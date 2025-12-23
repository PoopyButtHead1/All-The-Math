import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def get_monthly_log_returns(tickers, start="2015-01-01"):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    # month-end prices
    px_m = px.resample("M").last()
    # monthly log returns
    lr = np.log(px_m / px_m.shift(1)).dropna()
    # drop any columns that are still missing too much
    lr = lr.dropna(axis=1, how="any")
    return lr

def estimate_mu_cov_from_history(log_returns):
    mu_m = log_returns.mean().values          # monthly mean log return
    cov_m = log_returns.cov().values          # monthly covariance of log returns
    return mu_m, cov_m

# ============================================================
# 1. ETF Ticker Weights (Total Portfolio Weights)
# ============================================================

ETF_WEIGHTS = {
    # Identity (20%)
    "V":    0.06,
    "MA":   0.05,
    "OKTA": 0.04,
    "EFX":  0.03,
    "NTRS": 0.02,

    # Yield base (40%)
    "SGOV": 0.40,

    # Settlement (25%)
    "ICE":  0.07,
    "CME":  0.06,
    "FI":   0.04,
    "FIS":  0.04,
    "BR":   0.04,

    # Convex + oracle truth (15%)
    "IBIT": 0.05,
    "ETHA": 0.05,
    "COIN": 0.03,
    "SPGI": 0.02,
}

TICKERS = list(ETF_WEIGHTS.keys())
W = np.array([ETF_WEIGHTS[t] for t in TICKERS], dtype=float)
W = W / W.sum()


# Map each ticker to a sleeve
# Sleeves: identity, rwa, settlement, oracle
TICKER_TO_SLEEVE = {
    "V":"identity","MA":"identity","OKTA":"identity","EFX":"identity","NTRS":"identity",
    "SGOV":"rwa",
    "ICE":"settlement","CME":"settlement","FI":"settlement","FIS":"settlement","BR":"settlement",
    "IBIT":"oracle","ETHA":"oracle","COIN":"oracle","SPGI":"oracle",
}



SLEEVES = ["identity", "rwa", "settlement", "oracle"]
SLEEVE_INDEX = {s: i for i, s in enumerate(SLEEVES)}

# ============================================================
# 2. Derive Sleeve-Level Target Weights from ETF_WEIGHTS
# ============================================================

def compute_sleeve_targets(etf_weights, ticker_to_sleeve):
    sleeve_weights = {s: 0.0 for s in SLEEVES}
    for ticker, w in etf_weights.items():
        sleeve = ticker_to_sleeve[ticker]
        sleeve_weights[sleeve] += w
    # Normalize in case of rounding
    total = sum(sleeve_weights.values())
    for s in sleeve_weights:
        sleeve_weights[s] /= total
    return sleeve_weights

SLEEVE_TARGET_WEIGHTS = compute_sleeve_targets(ETF_WEIGHTS, TICKER_TO_SLEEVE)

print("Sleeve target weights:")
for s in SLEEVES:
    print(f"  {s}: {SLEEVE_TARGET_WEIGHTS[s]:.3f}")

# Also compute normalized weights *within* each sleeve (for attribution later)
def compute_within_sleeve_weights(etf_weights, ticker_to_sleeve):
    within = {s: {} for s in SLEEVES}
    totals = {s: 0.0 for s in SLEEVES}
    for t, w in etf_weights.items():
        sleeve = ticker_to_sleeve[t]
        totals[sleeve] += w
    for t, w in etf_weights.items():
        sleeve = ticker_to_sleeve[t]
        if totals[sleeve] > 0:
            within[sleeve][t] = w / totals[sleeve]
    return within

WITHIN_SLEEVE_WEIGHTS = compute_within_sleeve_weights(ETF_WEIGHTS, TICKER_TO_SLEEVE)


# ============================================================
# 3. Sleeve Return Parameters & Correlations
# ============================================================
# Annualized *baseline* expected returns per sleeve (normal regime)
# These are high-level, tweakable assumptions.
SLEEVE_MU_ANNUAL_REGIME = {
    "identity":   [-0.10, 0.10, 0.18],
    "rwa":        [-0.12, 0.10, 0.20],
    "settlement": [-0.10, 0.09, 0.16],
    "oracle":     [-0.12, 0.11, 0.20],
}


# Annualized volatilities per sleeve
SLEEVE_SIGMA_ANNUAL = {
    "identity":   0.25,  # growth/fintech + mega-cap mix
    "rwa":        0.20,  # alt managers / asset infra
    "settlement": 0.22,  # exchanges + payments processors
    "oracle":     0.24,  # data/software mix
}


# Regime-specific adjustments to expected returns (annual)
# Order of regimes: [bear, normal, bull]
SLEEVE_MU_ANNUAL_REGIME = {
    "identity":   [-0.05, 0.10, 0.18],
    "rwa":        [ 0.01, 0.04, 0.06],
    "settlement": [-0.20, 0.18, 0.35],
    "oracle":     [-0.25, 0.20, 0.40],
}

# Correlation matrix between sleeves (identity, rwa, settlement, oracle)
CORR_MATRIX = np.array([
    [1.00, 0.55, 0.65, 0.60],  # identity
    [0.55, 1.00, 0.55, 0.50],  # rwa
    [0.65, 0.55, 1.00, 0.55],  # settlement
    [0.60, 0.50, 0.55, 1.00],  # oracle
])


# ============================================================
# 4. Regime Switching Setup
# ============================================================

# Regimes: 0=bear, 1=normal, 2=bull
REGIME_NAMES = {0: "bear", 1: "normal", 2: "bull"}
N_REGIMES = 3

# Transition matrix: rows sum to 1
REGIME_TRANSITION = np.array([
    [0.80, 0.18, 0.02],  # bear -> (bear, normal, bull)
    [0.10, 0.80, 0.10],  # normal -> ...
    [0.02, 0.18, 0.80],  # bull -> ...
])


# ============================================================
# 5. Core 4-Sleeve Monte Carlo Engine
# ============================================================

def simulate_ticker_portfolio(
    log_returns_hist,          # DataFrame of monthly log returns for tickers
    etf_weights,               # dict ticker->weight
    n_paths=5000,
    years=20,
    steps_per_year=12,
    df_t=8,
    rebalance="monthly",       # "monthly" or "quarterly" or None
    start_value=100.0,
    seed=42,
):
    np.random.seed(seed)

    tickers = list(etf_weights.keys())
    w = np.array([etf_weights[t] for t in tickers], dtype=float)
    w = w / w.sum()

    # align history to tickers in weights (and drop missing tickers if needed)
    lr = log_returns_hist[tickers].dropna()

    mu_m, cov_m = estimate_mu_cov_from_history(lr)

    n_assets = len(tickers)
    n_steps = years * steps_per_year

    # Cholesky of covariance (not correlation)
    L = np.linalg.cholesky(cov_m)

    # helper: draw one multivariate t shock with covariance cov_m
    def draw_mv_t():
        z = np.random.normal(size=n_assets)
        chi2 = np.random.chisquare(df_t)
        scale = np.sqrt(df_t / chi2)
        # correlate using covariance structure
        return (L @ z) * scale

    portfolio_values = np.zeros((n_steps + 1, n_paths))
    portfolio_values[0, :] = start_value

    # optional: store holdings (so rebalancing is explicit)
    # holdings are dollar allocations per asset
    holdings = np.zeros((n_assets, n_paths))
    holdings[:, :] = (start_value * w)[:, None]

    # rebalance schedule
    if rebalance == "monthly":
        rebalance_every = 1
    elif rebalance == "quarterly":
        rebalance_every = 3
    else:
        rebalance_every = None

    for p in range(n_paths):
        for t in range(1, n_steps + 1):
            # draw monthly log returns: drift + shock
            # (mu_m is monthly mean log return)
            shock = draw_mv_t()
            step_lr = mu_m + shock

            # update holdings by applying exp(log return)
            holdings[:, p] = holdings[:, p] * np.exp(step_lr)

            total_val = holdings[:, p].sum()
            portfolio_values[t, p] = total_val

            # rebalance to target weights on schedule
            if rebalance_every is not None and (t % rebalance_every == 0) and total_val > 0:
                holdings[:, p] = total_val * w

    time_index = np.arange(n_steps + 1)
    portfolio_values_df = pd.DataFrame(
        portfolio_values,
        index=time_index,
        columns=[f"path_{i}" for i in range(n_paths)]
    )

    final_vals = portfolio_values_df.iloc[-1].values
    percentiles = np.percentile(final_vals, [1, 5, 25, 50, 75, 95, 99])

    summary = {
        "mean_final": float(final_vals.mean()),
        "median_final": float(np.median(final_vals)),
        "stdev_final": float(final_vals.std()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }

    ev_path = portfolio_values_df.mean(axis=1)

    return portfolio_values_df, summary, ev_path



# ============================================================
# 6. Example run
# ============================================================

if __name__ == "__main__":
    hist_lr = get_monthly_log_returns(TICKERS, start="2018-01-01")
    # IMPORTANT: hist_lr columns order must match ETF_WEIGHTS keys used by the simulator
    portfolio_vals, summary, ev_path = simulate_ticker_portfolio(
        log_returns_hist=hist_lr,
        etf_weights=ETF_WEIGHTS,
        n_paths=5000,
        years=20,
        steps_per_year=12,
        df_t=8,
        rebalance="quarterly",   # choose "monthly" or "quarterly"
        start_value=100.0,
        seed=42,
    )

    print("\n=== Ticker-Level Monte Carlo Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v:,.2f}")


# ============================================================
    # RISK METRICS SECTION
    # ============================================================

    # 1. Compute monthly returns for each path
    portfolio_array = portfolio_vals.values  # shape (time, paths)
    monthly_returns = (portfolio_array[1:, :] / portfolio_array[:-1, :]) - 1.0

    # 2. Annualization factor
    steps_per_year = 12

    # ------------------------------------------------------------
    # A. Annualized Volatility
    # ------------------------------------------------------------
    vol_monthly = np.std(monthly_returns, axis=0)  # stdev per path
    vol_annualized = vol_monthly * np.sqrt(steps_per_year)
    vol_final = vol_annualized.mean()

    # ------------------------------------------------------------
    # B. Sharpe Ratio (using 0% risk-free rate)
    # ------------------------------------------------------------
    mean_monthly_ret = np.mean(monthly_returns, axis=0)
    mean_annual_ret = (1 + mean_monthly_ret)**steps_per_year - 1
    sharpe = mean_annual_ret.mean() / vol_final

    # ------------------------------------------------------------
        # C. Sortino Ratio
    # ------------------------------------------------------------
    downside_returns = np.where(monthly_returns < 0, monthly_returns, np.nan)
    downside_dev_monthly = np.nanstd(downside_returns, axis=0)
    downside_dev_annual = downside_dev_monthly * np.sqrt(steps_per_year)
    sortino = mean_annual_ret.mean() / downside_dev_annual.mean()

    # ------------------------------------------------------------
    # D. Maximum Drawdown
    # ------------------------------------------------------------
    def compute_max_drawdown(path):
        cummax = np.maximum.accumulate(path)
        drawdown = (path - cummax) / cummax
        return drawdown.min()

    max_drawdowns = np.array([compute_max_drawdown(portfolio_array[:, i]) for i in range(portfolio_array.shape[1])])
    max_drawdown_avg = max_drawdowns.mean()

    # ------------------------------------------------------------
    # E. VaR and CVaR at 95%
    # ------------------------------------------------------------
    portfolio_array = portfolio_vals.values
    monthly_returns = (portfolio_array[1:, :] / portfolio_array[:-1, :]) - 1.0

    # 95% 1-month VaR/CVaR (return-based)
    VaR95 = np.percentile(monthly_returns, 5)
    CVaR95 = monthly_returns[monthly_returns <= VaR95].mean()


    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------
    print("\n=== RISK METRICS ===")
    print(f"Annualized Volatility: {vol_final:,.2f}")
    print(f"Sharpe Ratio: {sharpe:,.3f}")
    print(f"Sortino Ratio: {sortino:,.3f}")
    print(f"Average Max Drawdown: {max_drawdown_avg:.2%}")
    print(f"95% VaR: {VaR95:,.2f}")
    print(f"95% CVaR: {CVaR95:,.2f}")


    # ---------- Common helpers ----------
    time = portfolio_vals.index.values           # 0..240 months
    final_vals = portfolio_vals.iloc[-1].values  # shape (n_paths,)
    median_path = portfolio_vals.median(axis=1)  # median over paths

    # ============================================================
    # Plot 1: Expected value vs median path
    # ============================================================
    plt.figure(figsize=(8, 5))
    plt.plot(time, ev_path.values, label="Expected value (mean)")
    plt.plot(time, median_path.values, label="Median path", linestyle="--")
    plt.xlabel("Month")
    plt.ylabel("Portfolio value")
    plt.title("Expected vs Median Portfolio Path")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # Plot 2: Fan chart (5/25/50/75/95 percentiles)
    # ============================================================
    percentiles = [5, 25, 50, 75, 95]
    pct_paths = np.percentile(portfolio_vals.values, percentiles, axis=1)  # shape (len(pct), n_steps+1)

    plt.figure(figsize=(8, 5))
    # 5–95 band
    plt.fill_between(time, pct_paths[0, :], pct_paths[4, :], alpha=0.2, label="5–95% band")
    # 25–75 band
    plt.fill_between(time, pct_paths[1, :], pct_paths[3, :], alpha=0.3, label="25–75% band")
    # Median line
    plt.plot(time, pct_paths[2, :], label="Median (50%)", linewidth=2)
    plt.xlabel("Month")
    plt.ylabel("Portfolio value")
    plt.title("Monte Carlo Fan Chart (Percentile Bands)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # Plot 3: Histogram of final values (log scale on x)
    # ============================================================
    # To avoid issues with log(0), add a tiny epsilon
    eps = 1e-6
    log_final = np.log(final_vals + eps)

    plt.figure(figsize=(8, 5))
    plt.hist(log_final, bins=60)
    plt.xlabel("log(Final value)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Final Portfolio Values (log scale)")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # Plot 4: Average sleeve values over time
    # ============================================================
    plt.figure(figsize=(8, 5))
    for s in SLEEVES:
        # mean across paths at each time step
        mean_sleeve = portfolio_vals[s].mean(axis=1)
        plt.plot(time, mean_sleeve.values, label=s)
    plt.xlabel("Month")
    plt.ylabel("Average sleeve value")
    plt.title("Average Sleeve Values Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


