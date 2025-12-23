# onch_backtest.py
from __future__ import annotations

import math
import os
import time
import glob
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================
# Caching system
# ============================

CACHE_DIR = "data_cache"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def get_cache_path(ticker: str, start: str, end: str, adjusted: bool = True) -> str:
    """Generate cache file path for a ticker."""
    adj_suffix = "_adj" if adjusted else ""
    filename = f"{ticker}_{start}_{end}{adj_suffix}.csv"
    return os.path.join(CACHE_DIR, filename)


def load_from_cache(ticker: str, start: str, end: str, adjusted: bool = True) -> Optional[pd.Series]:
    """Load data from cache if it exists."""
    cache_path = get_cache_path(ticker, start, end, adjusted)
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"📦 Loading {ticker} from cache", end=" ", flush=True)
            series = df.iloc[:, 0] if df.shape[1] > 0 else df[ticker]
            print(f"✓ ({len(series)} days)")
            return series
        except Exception as e:
            print(f"⚠️  Cache load failed for {ticker}: {e}")
            return None
    return None


def save_to_cache(series: pd.Series, ticker: str, start: str, end: str, adjusted: bool = True) -> None:
    """Save data to cache."""
    ensure_cache_dir()
    cache_path = get_cache_path(ticker, start, end, adjusted)
    df = pd.DataFrame({ticker: series})
    df.to_csv(cache_path)


def clear_cache(ticker: Optional[str] = None) -> None:
    """Clear cache for specific ticker or all."""
    if ticker:
        pattern = os.path.join(CACHE_DIR, f"{ticker}_*.csv")
        files = glob.glob(pattern)
        for f in files:
            os.remove(f)
            print(f"  Cleared {f}")
    else:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print(f"  Cleared entire cache directory")


def get_cache_status() -> Dict[str, int]:
    """Return stats about cached files."""
    ensure_cache_dir()
    files = glob.glob(os.path.join(CACHE_DIR, "*.csv"))
    return {
        "num_files": len(files),
        "cache_dir": CACHE_DIR,
        "files": [os.path.basename(f) for f in files],
    }


# ============================
# 0) Portfolio definition
# ============================

# IMPORTANT:
# - Polygon uses "SQ" for Block (not "XYZ").
# - If you want to keep internal/branding tickers (e.g., "XYZ"), use SYMBOL_ALIASES below.
ONCH_WEIGHTS: Dict[str, float] = {
    "BLK": 0.07,
    "BK": 0.06,
    "STT": 0.05,
    "BEN": 0.04,
    "WT": 0.03,
    "SGOV": 0.05,

    "COIN": 0.075,
    "CME": 0.06,
    "SQ": 0.055,     # Block
    "PYPL": 0.09,
    "ICE": 0.06,

    # miners optional; include if you want
    # "RIOT": 0.06,

    "HOOD": 0.06,
    "SOFI": 0.05,
    "MSTR": 0.04,

    "OKTA": 0.04,
    "CRWD": 0.04,
    "PLTR": 0.035,
    "ACN": 0.025,
    "TRI": 0.01,
}

DEFAULT_BENCHMARK = "QQQ"

# If you *really* want to keep "XYZ" in your weights, do it like this:
# ONCH_WEIGHTS["XYZ"] = ONCH_WEIGHTS.pop("SQ")
# and set SYMBOL_ALIASES = {"XYZ": "SQ"}
SYMBOL_ALIASES: Dict[str, str] = {
    # "XYZ": "SQ",  # example alias: internal "XYZ" -> market "SQ"
}

# Prefer env var (safer) but will fall back if you hardcode
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip() or  "pBcpmvWGRDlptssSRzgfjrtcLUfqZ3Mn"


# ============================
# 1) Utilities
# ============================

def assert_weights_sum_to_one(weights: Dict[str, float], tol: float = 1e-8) -> None:
    s = float(sum(weights.values()))
    if abs(s - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1.0. Got {s:.10f}")

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(weights.values()))
    if s <= 0:
        raise ValueError("Weight sum must be > 0")
    return {k: float(v) / s for k, v in weights.items()}

def first_trading_day_of_month(index: pd.DatetimeIndex, month: int, year: int) -> Optional[pd.Timestamp]:
    mask = (index.year == year) & (index.month == month)
    dates = index[mask]
    if len(dates) == 0:
        return None
    return pd.Timestamp(dates.min())

def make_rebalance_dates(index: pd.DatetimeIndex, months: List[int]) -> List[pd.Timestamp]:
    years = sorted(set(index.year))
    out: List[pd.Timestamp] = []
    for y in years:
        for m in months:
            d = first_trading_day_of_month(index, m, y)
            if d is not None:
                out.append(d)
    out = [d for d in out if d >= index.min() and d <= index.max()]
    return sorted(set(out))

def annualize_cagr(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    n = len(r)
    if n == 0:
        return np.nan
    total = float((1.0 + r).prod())
    return total ** (periods_per_year / n) - 1.0

def annualize_vol(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))

def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan

def drawdown_duration(nav: pd.Series) -> Tuple[int, int]:
    """
    Returns (max_duration_days, current_duration_days) measured in trading days.
    """
    if len(nav) == 0:
        return 0, 0
    peak = nav.cummax()
    underwater = nav < peak
    # Compute consecutive runs of True
    durations = []
    cur = 0
    for u in underwater:
        if u:
            cur += 1
        else:
            durations.append(cur)
            cur = 0
    durations.append(cur)
    return int(max(durations)), int(cur)

def downside_dev(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    neg = r[r < 0]
    if len(neg) < 2:
        return 0.0
    return float(neg.std(ddof=1) * math.sqrt(periods_per_year))

def tracking_error(port_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 252) -> float:
    df = pd.concat([port_ret.rename("p"), bench_ret.rename("b")], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    active = df["p"] - df["b"]
    return float(active.std(ddof=1) * math.sqrt(periods_per_year))

def beta_alpha_corr(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> Tuple[float, float, float]:
    """
    Returns (beta, alpha_annual, corr).
    alpha computed as annualized intercept of daily regression:
      (Rp - Rf) = alpha_d + beta*(Rb - Rf) + eps
    alpha_annual = (1+alpha_d)^252 - 1 approx alpha_d*252 for small values
    """
    df = pd.concat([port_ret.rename("p"), bench_ret.rename("b")], axis=1).dropna()
    if len(df) < 10:
        return np.nan, np.nan, np.nan

    rf_d = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    y = (df["p"] - rf_d).to_numpy()
    x = (df["b"] - rf_d).to_numpy()

    x_var = np.var(x, ddof=1)
    if x_var <= 0:
        return np.nan, np.nan, np.nan

    beta = float(np.cov(x, y, ddof=1)[0, 1] / x_var)
    alpha_d = float(np.mean(y) - beta * np.mean(x))
    alpha_ann = float((1.0 + alpha_d) ** periods_per_year - 1.0)

    corr = float(np.corrcoef(df["p"], df["b"])[0, 1])
    return beta, alpha_ann, corr

def calmar_ratio(cagr: float, mdd: float) -> float:
    if mdd is None or np.isnan(mdd) or mdd >= 0:
        return np.nan
    return float(cagr / abs(mdd))

def hit_rate(port_ret: pd.Series, bench_ret: Optional[pd.Series] = None) -> float:
    if bench_ret is None:
        r = port_ret.dropna()
        return float((r > 0).mean()) if len(r) else np.nan
    df = pd.concat([port_ret.rename("p"), bench_ret.rename("b")], axis=1).dropna()
    if len(df) == 0:
        return np.nan
    return float((df["p"] > df["b"]).mean())

def capture_ratios(port_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float]:
    """
    Upside capture: sum(Rp | Rb>0) / sum(Rb | Rb>0)
    Downside capture: sum(Rp | Rb<0) / sum(Rb | Rb<0)
    """
    df = pd.concat([port_ret.rename("p"), bench_ret.rename("b")], axis=1).dropna()
    if len(df) < 10:
        return np.nan, np.nan

    up = df[df["b"] > 0]
    down = df[df["b"] < 0]

    def safe_ratio(a: pd.Series, b: pd.Series) -> float:
        denom = float(b.sum())
        return float(a.sum() / denom) if abs(denom) > 1e-12 else np.nan

    up_cap = safe_ratio(up["p"], up["b"]) if len(up) else np.nan
    down_cap = safe_ratio(down["p"], down["b"]) if len(down) else np.nan
    return up_cap, down_cap

def to_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    r = daily_returns.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    nav = (1.0 + r).cumprod()
    m = nav.resample("M").last().pct_change().dropna()
    m.index = m.index.to_period("M").to_timestamp("M")
    return m

def calendar_year_returns(daily_returns: pd.Series) -> pd.Series:
    r = daily_returns.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    nav = (1.0 + r).cumprod()
    y = nav.resample("Y").last().pct_change().dropna()
    y.index = y.index.year
    return y

def rolling_stats(daily_returns: pd.Series, window: int = 252, rf_annual: float = 0.0) -> pd.DataFrame:
    r = daily_returns.dropna()
    if len(r) < window:
        return pd.DataFrame(index=r.index, columns=["RollCAGR", "RollVol", "RollSharpe"], dtype=float)

    rf_d = (1.0 + rf_annual) ** (1.0 / 252) - 1.0

    nav = (1.0 + r).cumprod()
    # rolling CAGR from nav(t)/nav(t-window)
    roll_total = nav / nav.shift(window)
    roll_cagr = roll_total ** (252.0 / window) - 1.0
    roll_vol = r.rolling(window).std(ddof=1) * math.sqrt(252)

    roll_excess = r - rf_d
    roll_sharpe = (roll_excess.rolling(window).mean() * 252) / roll_vol

    return pd.DataFrame(
        {"RollCAGR": roll_cagr, "RollVol": roll_vol, "RollSharpe": roll_sharpe},
        index=r.index,
    )


# ============================
# 2) Polygon loaders
# ============================

def _map_to_polygon_symbol(ticker: str) -> str:
    return SYMBOL_ALIASES.get(ticker, ticker)

def polygon_get_adjusted_closes(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    adjusted: bool = True,
    max_retries: int = 5,
) -> pd.Series:
    """
    Fetch adjusted close prices from Polygon.io for a single ticker.
    Uses cache if available; downloads and caches if not.
    Returns Series indexed by Date (UTC-normalized to date boundary).
    """
    # Try cache first
    cached = load_from_cache(ticker, start, end, adjusted)
    if cached is not None:
        return cached

    if not api_key or api_key == "PUT_YOUR_KEY_HERE":
        raise ValueError("Polygon API key missing. Set POLYGON_API_KEY env var or paste your key.")

    import requests

    poly_ticker = _map_to_polygon_symbol(ticker)
    url = f"https://api.polygon.io/v2/aggs/ticker/{poly_ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    print(f"Fetching {ticker} (Polygon: {poly_ticker})...", end=" ", flush=True)

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                # rate limit
                wait = 12.0 + 3.0 * attempt
                print(f"(429 rate limit, sleep {wait:.0f}s)...", end=" ", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", None)
            if not results:
                print("(no data)")
                raise ValueError(f"No data returned for {ticker} (Polygon: {poly_ticker})")

            df = pd.DataFrame(
                {
                    "Date": pd.to_datetime([row["t"] for row in results], unit="ms").tz_localize("UTC").tz_convert(None),
                    ticker: [float(row["c"]) for row in results],
                }
            ).set_index("Date").sort_index()

            # Polygon timestamps are market close; normalize to date (keeps index as datetime)
            df.index = pd.to_datetime(df.index.date)

            s = df[ticker].astype(float)
            print(f"✓ ({len(s)} days)")
            
            # Save to cache
            save_to_cache(s, ticker, start, end, adjusted)
            print(f"💾 Cached {ticker}", flush=True)
            return s

        except Exception as e:
            last_err = e
            wait = 5.0 * (2 ** attempt)
            print(f"(error, retry in {wait:.0f}s)...", end=" ", flush=True)
            time.sleep(wait)

    print("✗")
    raise RuntimeError(f"Failed to fetch {ticker} after {max_retries} retries: {last_err}")

def polygon_price_panel(
    tickers: List[str],
    start: str,
    end: str,
    api_key: str,
    adjusted: bool = True,
    delay_between_requests: float = 12.0,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for multiple tickers.
    Uses cache if available; downloads and caches if not.
    delay_between_requests=12 sec keeps you within Polygon free tier (≈5/min).
    """
    series = []
    for i, t in enumerate(tickers):
        try:
            s = polygon_get_adjusted_closes(t, start, end, api_key, adjusted=adjusted)
            series.append(s.rename(t))
        except Exception as e:
            print(f"\n  Skipping {t}: {e}")
        if i < len(tickers) - 1:
            time.sleep(delay_between_requests)

    if not series:
        raise ValueError("No tickers downloaded successfully.")

    prices = pd.concat(series, axis=1).sort_index()
    return prices


# ============================
# 3) Backtest engine
# ============================

@dataclass
class BacktestConfig:
    weights: Dict[str, float]
    benchmark: str = DEFAULT_BENCHMARK

    # Schedules
    rebalance_months: Tuple[int, ...] = (1, 4, 7, 10)  # quarterly
    reconstitution_months: Tuple[int, ...] = (1, 7)    # semi-annual hook

    # Costs and assumptions
    trading_cost_bps: float = 10.0  # applied on rebalance turnover
    mgmt_fee_bps_annual: float = 0.0  # optional: simulate ETF fee drag
    rf_annual: float = 0.0

    require_full_history: bool = True  # start only when all tickers have data
    periods_per_year: int = 252

Reconstituter = Callable[[pd.Timestamp, List[str]], List[str]]

def run_backtest(
    prices: pd.DataFrame,
    bench_prices: pd.Series,
    cfg: BacktestConfig,
    reconstituter: Optional[Reconstituter] = None,
) -> Dict[str, object]:
    weights = normalize_weights(cfg.weights)
    assert_weights_sum_to_one(weights)

    tickers = list(weights.keys())
    missing_cols = [t for t in tickers if t not in prices.columns]
    if missing_cols:
        raise ValueError(f"Missing tickers in prices: {missing_cols}")

    panel = prices[tickers].copy().sort_index()
    bench_prices = bench_prices.copy().sort_index()

    # Select window
    if cfg.require_full_history:
        panel = panel.dropna(how="any")
    else:
        panel = panel.dropna(how="all")

    if len(panel) < 60:
        raise ValueError("Not enough overlapping data. Provide more history or set require_full_history=False.")

    # Daily returns
    rets = panel.pct_change().dropna(how="any")
    bench_ret = bench_prices.pct_change().dropna()

    # Align on common dates AFTER return calc (critical fix)
    common_idx = rets.index.intersection(bench_ret.index)
    rets = rets.reindex(common_idx).dropna(how="any")
    bench_ret = bench_ret.reindex(common_idx).dropna()

    if len(common_idx) < 60:
        raise ValueError("Not enough common return dates between portfolio and benchmark.")

    # Rebalance dates on the aligned return index
    rebalance_dates = make_rebalance_dates(rets.index, list(cfg.rebalance_months))
    reconst_dates = make_rebalance_dates(rets.index, list(cfg.reconstitution_months))

    # Weight vectors
    w_target = pd.Series(weights, index=rets.columns, dtype=float)
    w_current = w_target.copy()

    nav_level = 1.0
    nav_list = []
    port_ret_list = []

    turnover_rows = []
    last_w = w_current.copy()

    # Daily fee drag (optional)
    fee_d = (cfg.mgmt_fee_bps_annual / 10000.0) / cfg.periods_per_year

    for d in rets.index:
        # optional reconstitution hook
        if (reconstituter is not None) and (d in reconst_dates):
            current_universe = list(w_current.index)
            new_universe = reconstituter(d, current_universe)
            if set(new_universe) != set(current_universe):
                raise NotImplementedError("Universe changes require updating weights and price panel.")

        # rebalance
        if d in rebalance_dates:
            w_current = w_target.copy()

            turnover = float((w_current - last_w).abs().sum() / 2.0)  # one-way turnover
            trade_cost = turnover * (cfg.trading_cost_bps / 10000.0)
            nav_level *= (1.0 - trade_cost)

            turnover_rows.append((d, turnover, trade_cost))
            last_w = w_current.copy()

        # daily return from holdings
        r = float((w_current * rets.loc[d]).sum())

        # apply management fee drag daily (approx)
        r_after_fee = r - fee_d

        nav_level *= (1.0 + r_after_fee)

        port_ret_list.append((d, r_after_fee))
        nav_list.append((d, nav_level))

    port_ret = pd.Series(dict(port_ret_list), name="PortfolioReturn")
    nav = pd.Series(dict(nav_list), name="NAV")

    # Benchmark NAV on same dates
    bench_ret = bench_ret.reindex(port_ret.index)
    bench_nav = (1.0 + bench_ret).cumprod()
    bench_nav.name = "BenchmarkNAV"

    # Turnover / costs
    turnover_df = pd.DataFrame(turnover_rows, columns=["Date", "Turnover", "TradingCost"]).set_index("Date")
    avg_turnover = float(turnover_df["Turnover"].mean()) if len(turnover_df) else 0.0
    total_trade_cost = float(turnover_df["TradingCost"].sum()) if len(turnover_df) else 0.0

    # Annualized turnover: avg one-way per rebalance * rebalances/year
    rebalances_per_year = len(cfg.rebalance_months)
    annual_turnover_est = avg_turnover * rebalances_per_year

    # Performance stats
    cagr = annualize_cagr(port_ret, cfg.periods_per_year)
    vol = annualize_vol(port_ret, cfg.periods_per_year)
    rf = cfg.rf_annual
    sharpe = (cagr - rf) / vol if vol and vol > 0 else np.nan
    sortino_den = downside_dev(port_ret, cfg.periods_per_year)
    sortino = (cagr - rf) / sortino_den if sortino_den and sortino_den > 0 else np.nan

    mdd = max_drawdown(nav)
    dd_max_dur, dd_cur_dur = drawdown_duration(nav)

    beta, alpha_ann, corr = beta_alpha_corr(port_ret, bench_ret, rf_annual=rf, periods_per_year=cfg.periods_per_year)

    te = tracking_error(port_ret, bench_ret, cfg.periods_per_year)
    info_ratio = (cagr - annualize_cagr(bench_ret, cfg.periods_per_year)) / te if te and te > 0 else np.nan

    calmar = calmar_ratio(cagr, mdd)
    up_cap, down_cap = capture_ratios(port_ret, bench_ret)

    # Monthly / calendar returns
    port_m = to_monthly_returns(port_ret)
    bench_m = to_monthly_returns(bench_ret)
    port_y = calendar_year_returns(port_ret)
    bench_y = calendar_year_returns(bench_ret)

    best_day = float(port_ret.max()) if len(port_ret) else np.nan
    worst_day = float(port_ret.min()) if len(port_ret) else np.nan
    best_month = float(port_m.max()) if len(port_m) else np.nan
    worst_month = float(port_m.min()) if len(port_m) else np.nan

    # Hit rates
    hit_vs_zero = hit_rate(port_ret)
    hit_vs_bench = hit_rate(port_ret, bench_ret)

    # Rolling 12m stats
    rolling_12m = rolling_stats(port_ret, window=252, rf_annual=rf)

    # Return contribution (simple approximation from start->end)
    px2 = panel.reindex([port_ret.index.min(), port_ret.index.max()]).dropna(how="any")
    tot_by_ticker = (px2.iloc[-1] / px2.iloc[0] - 1.0)
    contrib = (w_target * tot_by_ticker).sort_values(ascending=False).rename("ReturnContributionApprox")

    # “Cost drag” reporting in bps/year (rough)
    n_days = len(port_ret.dropna())
    years = n_days / cfg.periods_per_year if n_days else np.nan
    cost_drag_bps_per_year = (total_trade_cost / years) * 10000.0 if years and years > 0 else 0.0

    stats = pd.Series(
        {
            "Start": port_ret.index.min(),
            "End": port_ret.index.max(),
            "Days": int(n_days),

            "TotalReturn": float(nav.iloc[-1] - 1.0),
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,

            "MaxDrawdown": mdd,
            "MaxDrawdownDurationDays": dd_max_dur,
            "CurrentDrawdownDurationDays": dd_cur_dur,
            "Calmar": calmar,

            "Beta_vs_Benchmark": beta,
            "Alpha_Annual_vs_Benchmark": alpha_ann,
            "Corr_vs_Benchmark": corr,

            "TrackingError": te,
            "InformationRatio": info_ratio,

            "UpsideCapture": up_cap,
            "DownsideCapture": down_cap,

            "HitRate_PositiveDays": hit_vs_zero,
            "HitRate_vs_BenchmarkDays": hit_vs_bench,

            "BestDay": best_day,
            "WorstDay": worst_day,
            "BestMonth": best_month,
            "WorstMonth": worst_month,

            "RebalanceCount": int(len([d for d in rebalance_dates if port_ret.index.min() <= d <= port_ret.index.max()])),
            "AvgOneWayTurnoverPerRebalance": avg_turnover,
            "AnnualizedTurnoverEst": annual_turnover_est,

            "TotalTradingCost_DecimalNAV": total_trade_cost,  # e.g. 0.02 = 2% NAV drag
            "TradingCostDrag_bps_per_year_est": cost_drag_bps_per_year,
            "MgmtFee_bps_annual": cfg.mgmt_fee_bps_annual,
        },
        name="Stats",
    )

    # Monthly returns table (portfolio, benchmark, active)
    monthly = pd.concat(
        [
            port_m.rename("Port_Monthly"),
            bench_m.rename("Bench_Monthly"),
        ],
        axis=1,
    )
    monthly["Active_Monthly"] = monthly["Port_Monthly"] - monthly["Bench_Monthly"]

    # Calendar year table
    yearly = pd.concat(
        [
            port_y.rename("Port_Year"),
            bench_y.rename("Bench_Year"),
        ],
        axis=1,
    )
    yearly["Active_Year"] = yearly["Port_Year"] - yearly["Bench_Year"]

    return {
        "nav": nav,
        "port_ret": port_ret,
        "bench_ret": bench_ret,
        "bench_nav": bench_nav,
        "stats": stats,
        "turnover": turnover_df,
        "contrib": contrib,
        "monthly_returns": monthly,
        "calendar_returns": yearly,
        "rolling_12m": rolling_12m,
        "rebalance_dates": rebalance_dates,
        "reconstitution_dates": reconst_dates,
    }


# ============================
# 4) Main
# ============================

if __name__ == "__main__":
    # If you still have legacy "XYZ" in your weights, map it here:
    # SYMBOL_ALIASES["XYZ"] = "SQ"
    # and put "XYZ" in ONCH_WEIGHTS instead of "SQ"

    WEIGHTS = normalize_weights(ONCH_WEIGHTS)
    assert_weights_sum_to_one(WEIGHTS)

    TICKERS = list(WEIGHTS.keys())
    BENCH = DEFAULT_BENCHMARK

    START = "2018-01-01"
    END = "2024-12-31"

    # Show cache status
    cache_status = get_cache_status()
    print("=" * 70)
    print("ONCH BACKTEST - Data Loading")
    print("=" * 70)
    print(f"\n📂 Cache Status: {cache_status['num_files']} files cached")
    if cache_status['files']:
        print(f"   To clear cache, run: clear_cache()")
    print()

    print("📥 Loading portfolio data (using cache if available)...")
    prices = polygon_price_panel(
        tickers=TICKERS,
        start=START,
        end=END,
        api_key=POLYGON_API_KEY,
        adjusted=True,
        delay_between_requests=12.0,  # 1 call / 12s
    )

    print("\n⏳ Waiting before benchmark download...")
    time.sleep(2)

    print("📥 Loading benchmark data (using cache if available)...")
    bench_prices = polygon_get_adjusted_closes(
        ticker=BENCH,
        start=START,
        end=END,
        api_key=POLYGON_API_KEY,
        adjusted=True,
    )

    print("\n" + "=" * 70)
    print("Running backtest...")
    print("=" * 70)

    # If any tickers failed download, rescale weights to available tickers
    available = [t for t in TICKERS if t in prices.columns and prices[t].notna().any()]
    missing = sorted(set(TICKERS) - set(available))
    if missing:
        print(f"\nWARNING: missing tickers removed and weights rescaled: {missing}")
        WEIGHTS = normalize_weights({t: WEIGHTS[t] for t in available})

    cfg = BacktestConfig(
        weights=WEIGHTS,
        benchmark=BENCH,
        trading_cost_bps=10.0,
        mgmt_fee_bps_annual=0.0,  # set e.g. 75 for 0.75% if you want fee drag
        rf_annual=0.0,
        require_full_history=True,
    )

    result = run_backtest(prices, bench_prices, cfg)

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (ETF / Index Factsheet Core)")
    print("=" * 70)
    print(result["stats"].to_string())

    print("\nTop return contributors (approx):")
    print(result["contrib"].head(12).to_string())

    print("\nMonthly returns (last 12):")
    print(result["monthly_returns"].tail(12).to_string())

    print("\nCalendar year returns:")
    print(result["calendar_returns"].to_string())

    print("\n✅ Backtest complete! Data has been cached for faster future runs.")
