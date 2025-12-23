# step5_stress_testing.py
# ============================================================
# STEP 5: Institutional Regime Stress Testing Battery
# Modules:
#   A) Crypto-led drawdowns
#   B) Equity macro stress (benchmark selloffs / vol spikes)
#   C) Correlation regime shifts (rolling corr buckets)
#   D) Volatility & drawdown envelope (rolling metrics & breach flags)
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List


# -------------------------
# Utilities
# -------------------------

def to_returns(x: pd.Series) -> pd.Series:
    """Accepts either returns or prices. If values look like prices, convert to pct_change."""
    x = x.dropna().copy()
    if x.empty:
        return x
    # Heuristic: if typical daily magnitude is > 0.5%? could still be returns.
    # Another: if series is mostly positive and large (e.g. 50-500), it's prices.
    if (x.min() > -1) and (x.max() > 5):  # likely price-like
        return x.pct_change().dropna()
    # If returns-like (bounded), keep
    return x.astype(float)


def align_series(*series: pd.Series) -> List[pd.Series]:
    """Align multiple series to common dates and drop any rows with NaNs."""
    s = [to_returns(x) for x in series if x is not None]
    if not s:
        return []
    df = pd.concat(s, axis=1).dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])]


def equity_curve(ret: pd.Series, start: float = 1.0) -> pd.Series:
    ret = ret.fillna(0.0)
    return start * (1.0 + ret).cumprod()


def max_drawdown_from_equity(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min()) if len(dd) else np.nan


def drawdown_series(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    return (eq / peak) - 1.0


def ann_vol(ret: pd.Series, periods_per_year: int = 252) -> float:
    return float(ret.std(ddof=1) * np.sqrt(periods_per_year)) if len(ret) else np.nan


def total_return(ret: pd.Series) -> float:
    eq = equity_curve(ret, 1.0)
    return float(eq.iloc[-1] - 1.0) if len(eq) else np.nan


def cagr(ret: pd.Series, periods_per_year: int = 252) -> float:
    if len(ret) < 2:
        return np.nan
    eq = equity_curve(ret, 1.0)
    years = len(ret) / periods_per_year
    if years <= 0:
        return np.nan
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0)


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    return a.rolling(window).corr(b)


def rolling_ann_vol(ret: pd.Series, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    return ret.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_max_drawdown(ret: pd.Series, window: int = 252) -> pd.Series:
    # Rolling max drawdown over trailing window via equity curve inside each window.
    # Efficient-ish: apply over windows.
    eq = equity_curve(ret, 1.0)
    def mdd(win: pd.Series) -> float:
        return max_drawdown_from_equity(win)
    # Need equity curve within window, not raw returns; use equity curve window
    # We'll compute dd on the windowed equity curve normalized to 1 at start.
    def mdd_eq(win_eq: pd.Series) -> float:
        norm = win_eq / win_eq.iloc[0]
        return max_drawdown_from_equity(norm)
    return eq.rolling(window).apply(mdd_eq, raw=False)


@dataclass
class EpisodeResult:
    EpisodeStart: pd.Timestamp
    EpisodeEnd: pd.Timestamp
    Days: int
    PortTotal: float
    BenchTotal: float
    PortMaxDD: float
    BenchMaxDD: float
    PortVol: float
    BenchVol: float
    PortCorrToBench: float
    HitRateVsBench: float


def summarize_episode(port_ret: pd.Series, bench_ret: pd.Series) -> Dict[str, float]:
    df = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if df.empty:
        return dict(
            Days=0, PortTotal=np.nan, BenchTotal=np.nan,
            PortMaxDD=np.nan, BenchMaxDD=np.nan,
            PortVol=np.nan, BenchVol=np.nan,
            PortCorrToBench=np.nan, HitRateVsBench=np.nan,
        )

    pr = df.iloc[:, 0]
    br = df.iloc[:, 1]

    port_eq = equity_curve(pr)
    bench_eq = equity_curve(br)

    port_dd = max_drawdown_from_equity(port_eq)
    bench_dd = max_drawdown_from_equity(bench_eq)

    hitrate = float((pr > br).mean())

    corr = float(pr.corr(br)) if len(pr) > 2 else np.nan

    return dict(
        Days=int(len(df)),
        PortTotal=total_return(pr),
        BenchTotal=total_return(br),
        PortMaxDD=port_dd,
        BenchMaxDD=bench_dd,
        PortVol=ann_vol(pr),
        BenchVol=ann_vol(br),
        PortCorrToBench=corr,
        HitRateVsBench=hitrate,
    )


# -------------------------
# A) Crypto-led drawdown episodes
# -------------------------

def find_crypto_led_drawdown_episodes(
    crypto_ret: pd.Series,
    threshold_dd: float = -0.30,
    min_days: int = 20,
    cooldown_days: int = 10,
) -> pd.DataFrame:
    """
    Identify episodes where crypto experiences a peak-to-trough drawdown <= threshold_dd.
    Returns episodes with start/end dates based on drawdown progression.
    """
    cr = to_returns(crypto_ret).dropna()
    if cr.empty:
        return pd.DataFrame(columns=["EpisodeStart", "EpisodeEnd", "CryptoMinDD", "CryptoTotal", "Days"])

    eq = equity_curve(cr, 1.0)
    dd = drawdown_series(eq)

    # Episodes: when dd crosses below threshold, start at prior peak date; end at recovery to 0 or end of series.
    episodes = []
    in_episode = False
    start = None
    trough_dd = 0.0
    trough_date = None
    peak_date = None

    peak_eq = eq.iloc[0]
    peak_date = eq.index[0]

    i = 0
    idx = eq.index

    while i < len(eq):
        date = idx[i]
        if eq.iloc[i] >= peak_eq:
            peak_eq = eq.iloc[i]
            peak_date = date

        current_dd = dd.iloc[i]

        if (not in_episode) and (current_dd <= threshold_dd):
            # Start at last peak
            in_episode = True
            start = peak_date
            trough_dd = current_dd
            trough_date = date

        if in_episode:
            if current_dd < trough_dd:
                trough_dd = current_dd
                trough_date = date

            # End condition: recovered to new high (dd ~ 0), OR end-of-series
            if current_dd >= -1e-6 and date > start:
                end = date
                # Store
                win = cr.loc[start:end]
                if len(win) >= min_days:
                    episodes.append({
                        "EpisodeStart": pd.Timestamp(start),
                        "EpisodeEnd": pd.Timestamp(end),
                        "CryptoMinDD": float(trough_dd),
                        "CryptoTotal": total_return(win),
                        "Days": int(len(win)),
                    })
                # cooldown: skip forward
                in_episode = False
                start = None
                # apply cooldown
                i += cooldown_days

        i += 1

    # If still in episode at end
    if in_episode and start is not None:
        end = idx[-1]
        win = cr.loc[start:end]
        if len(win) >= min_days:
            episodes.append({
                "EpisodeStart": pd.Timestamp(start),
                "EpisodeEnd": pd.Timestamp(end),
                "CryptoMinDD": float(trough_dd),
                "CryptoTotal": total_return(win),
                "Days": int(len(win)),
            })

    return pd.DataFrame(episodes).sort_values(["EpisodeStart", "EpisodeEnd"]).reset_index(drop=True)


def run_crypto_led_stress(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    crypto_ret: pd.Series,
    threshold_dd: float = -0.30,
    min_days: int = 20,
) -> pd.DataFrame:
    pr, br, cr = align_series(port_ret, bench_ret, crypto_ret)
    episodes = find_crypto_led_drawdown_episodes(cr, threshold_dd=threshold_dd, min_days=min_days)

    rows = []
    for _, ep in episodes.iterrows():
        s = ep["EpisodeStart"]
        e = ep["EpisodeEnd"]
        pwin = pr.loc[s:e]
        bwin = br.loc[s:e]

        stats = summarize_episode(pwin, bwin)
        rows.append({
            "EpisodeStart": s,
            "EpisodeEnd": e,
            "CryptoMinDD": ep["CryptoMinDD"],
            "CryptoTotal": ep["CryptoTotal"],
            **stats
        })
    return pd.DataFrame(rows)


# -------------------------
# B) Equity Macro Stress
# -------------------------

def find_equity_macro_episodes(
    bench_ret: pd.Series,
    selloff_dd: float = -0.10,
    min_days: int = 10,
    vol_window: int = 20,
    vol_spike_z: float = 2.0,
    cooldown_days: int = 10,
) -> pd.DataFrame:
    """
    Two triggers (either can qualify an episode):
      1) Benchmark drawdown <= selloff_dd
      2) Benchmark realized vol spike: rolling vol z-score >= vol_spike_z
    Episodes start at prior peak for dd-trigger; for vol-trigger start at vol crossing date.
    End when benchmark drawdown recovers to ~0 (dd trigger) OR vol normalizes (z < 1.0) or max 60 days.
    """
    br = to_returns(bench_ret).dropna()
    if br.empty:
        return pd.DataFrame(columns=["EpisodeStart", "EpisodeEnd", "Trigger", "BenchMinDD", "BenchTotal", "Days"])

    eq = equity_curve(br, 1.0)
    dd = drawdown_series(eq)

    rv = br.rolling(vol_window).std(ddof=1)
    z = (rv - rv.rolling(252).mean()) / rv.rolling(252).std(ddof=1)

    idx = br.index
    episodes = []
    in_ep = False
    start = None
    trigger = None
    peak_eq = eq.iloc[0]
    peak_date = eq.index[0]
    min_dd = 0.0

    i = 0
    while i < len(idx):
        date = idx[i]

        if eq.loc[date] >= peak_eq:
            peak_eq = eq.loc[date]
            peak_date = date

        dd_i = dd.loc[date]
        z_i = z.loc[date] if date in z.index else np.nan

        dd_trigger = (dd_i <= selloff_dd)
        vol_trigger = (pd.notna(z_i) and z_i >= vol_spike_z)

        if not in_ep and (dd_trigger or vol_trigger):
            in_ep = True
            if dd_trigger:
                start = peak_date
                trigger = "Drawdown"
                min_dd = dd_i
            else:
                start = date
                trigger = "VolSpike"
                min_dd = dd_i  # not the main metric but still record

        if in_ep:
            min_dd = min(min_dd, dd_i)

            # end logic
            max_len = 60
            elapsed = (date - start).days

            if trigger == "Drawdown":
                if dd_i >= -1e-6 and date > start:
                    end = date
                    win = br.loc[start:end]
                    if len(win) >= min_days:
                        episodes.append({
                            "EpisodeStart": pd.Timestamp(start),
                            "EpisodeEnd": pd.Timestamp(end),
                            "Trigger": trigger,
                            "BenchMinDD": float(min_dd),
                            "BenchTotal": total_return(win),
                            "Days": int(len(win)),
                        })
                    in_ep = False
                    i += cooldown_days
            else:  # VolSpike
                # normalize when z < 1 or time out
                if (pd.notna(z_i) and z_i < 1.0 and date > start) or elapsed >= max_len:
                    end = date
                    win = br.loc[start:end]
                    if len(win) >= min_days:
                        episodes.append({
                            "EpisodeStart": pd.Timestamp(start),
                            "EpisodeEnd": pd.Timestamp(end),
                            "Trigger": trigger,
                            "BenchMinDD": float(min_dd),
                            "BenchTotal": total_return(win),
                            "Days": int(len(win)),
                        })
                    in_ep = False
                    i += cooldown_days

        i += 1

    if in_ep and start is not None:
        end = idx[-1]
        win = br.loc[start:end]
        if len(win) >= min_days:
            episodes.append({
                "EpisodeStart": pd.Timestamp(start),
                "EpisodeEnd": pd.Timestamp(end),
                "Trigger": trigger,
                "BenchMinDD": float(min_dd),
                "BenchTotal": total_return(win),
                "Days": int(len(win)),
            })

    return pd.DataFrame(episodes).sort_values(["EpisodeStart", "EpisodeEnd"]).reset_index(drop=True)


def run_equity_macro_stress(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    selloff_dd: float = -0.10,
    vol_window: int = 20,
    vol_spike_z: float = 2.0,
    min_days: int = 10,
) -> pd.DataFrame:
    pr, br = align_series(port_ret, bench_ret)
    episodes = find_equity_macro_episodes(
        br, selloff_dd=selloff_dd, min_days=min_days,
        vol_window=vol_window, vol_spike_z=vol_spike_z
    )

    rows = []
    for _, ep in episodes.iterrows():
        s = ep["EpisodeStart"]
        e = ep["EpisodeEnd"]
        stats = summarize_episode(pr.loc[s:e], br.loc[s:e])
        rows.append({
            "EpisodeStart": s,
            "EpisodeEnd": e,
            "Trigger": ep["Trigger"],
            "BenchMinDD": ep["BenchMinDD"],
            "BenchTotal_Episode": ep["BenchTotal"],
            **stats
        })
    return pd.DataFrame(rows)


# -------------------------
# C) Correlation Regime Shifts
# -------------------------

def correlation_regime_table(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    corr_window: int = 63,
    low_q: float = 0.33,
    high_q: float = 0.67,
) -> pd.DataFrame:
    pr, br = align_series(port_ret, bench_ret)
    rc = rolling_corr(pr, br, corr_window).dropna()

    if rc.empty:
        return pd.DataFrame(columns=["Regime", "Days", "TotalReturn", "CAGR", "Vol", "MaxDD"])

    low = rc.quantile(low_q)
    high = rc.quantile(high_q)

    # label each day by corr regime
    regime = pd.Series(index=rc.index, dtype="object")
    regime.loc[rc <= low] = "LowCorr"
    regime.loc[(rc > low) & (rc < high)] = "MidCorr"
    regime.loc[rc >= high] = "HighCorr"

    # align returns to regime dates
    df = pd.concat([pr, br, regime], axis=1).dropna()
    df.columns = ["Port", "Bench", "Regime"]

    rows = []
    for name in ["LowCorr", "MidCorr", "HighCorr"]:
        sub = df[df["Regime"] == name]["Port"]
        if sub.empty:
            continue
        rows.append({
            "Regime": name,
            "Days": int(len(sub)),
            "TotalReturn": total_return(sub),
            "CAGR": cagr(sub),
            "Vol": ann_vol(sub),
            "MaxDD": max_drawdown_from_equity(equity_curve(sub)),
        })

    return pd.DataFrame(rows)


# -------------------------
# D) Volatility & Drawdown Envelope
# -------------------------

def vol_drawdown_envelope(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    window: int = 252,
    vol_mult_limit: float = 1.4,
    dd_gap_limit: float = 0.10,
) -> pd.DataFrame:
    """
    Produces rolling trailing window metrics for both portfolio and benchmark plus 'breach' flags:
      - VolBreach: port_roll_vol > vol_mult_limit * bench_roll_vol
      - DDBreach:  port_roll_mdd < (bench_roll_mdd - dd_gap_limit)  (i.e., worse by > gap)
    """
    pr, br = align_series(port_ret, bench_ret)

    out = pd.DataFrame(index=pr.index)
    out["Port_RollVol"] = rolling_ann_vol(pr, window=window)
    out["Bench_RollVol"] = rolling_ann_vol(br, window=window)

    out["Port_RollTotal"] = (1 + pr).rolling(window).apply(lambda x: np.prod(x) - 1.0, raw=True)
    out["Bench_RollTotal"] = (1 + br).rolling(window).apply(lambda x: np.prod(x) - 1.0, raw=True)

    out["Port_RollMaxDD"] = rolling_max_drawdown(pr, window=window)
    out["Bench_RollMaxDD"] = rolling_max_drawdown(br, window=window)

    out["VolBreach"] = out["Port_RollVol"] > (vol_mult_limit * out["Bench_RollVol"])
    out["DDBreach"] = out["Port_RollMaxDD"] < (out["Bench_RollMaxDD"] - dd_gap_limit)

    return out.dropna()


# -------------------------
# Orchestrator (one call)
# -------------------------

def run_step5_battery(
    port: pd.Series,
    bench: pd.Series,
    crypto: Optional[pd.Series] = None,
    crypto_dd_thresh: float = -0.30,
    equity_selloff_dd: float = -0.10,
    corr_window: int = 63,
    envelope_window: int = 252,
    out_prefix: str = "STEP5",
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of DataFrames + saves CSVs.
    If you want charts, keep these outputs and plot in your notebook/report layer.
    """
    results: Dict[str, pd.DataFrame] = {}

    # A
    if crypto is not None:
        df_a = run_crypto_led_stress(port, bench, crypto, threshold_dd=crypto_dd_thresh)
        results["A_crypto_led_drawdowns"] = df_a
        df_a.to_csv(f"{out_prefix}_A_crypto_led_drawdowns.csv", index=False)

    # B
    df_b = run_equity_macro_stress(port, bench, selloff_dd=equity_selloff_dd)
    results["B_equity_macro_stress"] = df_b
    df_b.to_csv(f"{out_prefix}_B_equity_macro_stress.csv", index=False)

    # C
    df_c = correlation_regime_table(port, bench, corr_window=corr_window)
    results["C_correlation_regimes"] = df_c
    df_c.to_csv(f"{out_prefix}_C_correlation_regimes.csv", index=False)

    # D
    df_d = vol_drawdown_envelope(port, bench, window=envelope_window)
    results["D_vol_dd_envelope"] = df_d
    df_d.to_csv(f"{out_prefix}_D_vol_dd_envelope.csv")

    return results


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    """
    Integration with your ONCH_Backtest.py:
    Runs financial stress tests on the portfolio backtest results.
    """
    import time
    from Backtest import (
        polygon_get_adjusted_closes,
        polygon_price_panel,
        run_backtest,
        BacktestConfig,
        ONCH_WEIGHTS,
        normalize_weights,
        POLYGON_API_KEY,
        DEFAULT_BENCHMARK,
    )

    # Configuration
    START = "2018-01-01"
    END = "2024-12-31"

    print("=" * 70)
    print("ETF FINANCIAL STRESS TEST SUITE")
    print("=" * 70)

    # Step 1: Download backtest data
    WEIGHTS = normalize_weights(ONCH_WEIGHTS)
    TICKERS = list(WEIGHTS.keys())
    BENCH = DEFAULT_BENCHMARK

    print("\nDownloading portfolio data from Polygon.io...")
    prices = polygon_price_panel(
        tickers=TICKERS,
        start=START,
        end=END,
        api_key=POLYGON_API_KEY,
        adjusted=True,
        delay_between_requests=12.0,
    )

    print("Waiting before benchmark download...")
    time.sleep(5)

    bench_prices = polygon_get_adjusted_closes(
        ticker=BENCH,
        start=START,
        end=END,
        api_key=POLYGON_API_KEY,
        adjusted=True,
    )

    # Step 2: Run backtest
    print("\nRunning backtest...")
    available = [t for t in TICKERS if t in prices.columns and prices[t].notna().any()]
    WEIGHTS = normalize_weights({t: WEIGHTS[t] for t in available})

    cfg = BacktestConfig(
        weights=WEIGHTS,
        benchmark=BENCH,
        trading_cost_bps=10.0,
        mgmt_fee_bps_annual=0.0,
        rf_annual=0.0,
        require_full_history=True,
    )

    result = run_backtest(prices, bench_prices, cfg)

    # Step 3: Download crypto proxy for stress testing
    print("\nDownloading crypto proxy (COIN - for crypto stress tests)...")
    time.sleep(5)
    coin_px = polygon_get_adjusted_closes(
        ticker="COIN",
        start=START,
        end=END,
        api_key=POLYGON_API_KEY,
        adjusted=True,
    )
    crypto_ret = coin_px.pct_change().dropna()

    # Step 4: Run stress test battery
    print("\n" + "=" * 70)
    print("Running financial stress tests...")
    print("=" * 70)

    results = run_step5_battery(
        port=result["port_ret"],
        bench=result["bench_ret"],
        crypto=crypto_ret,
        crypto_dd_thresh=-0.30,
        equity_selloff_dd=-0.10,
        corr_window=63,
        envelope_window=252,
        out_prefix="STRESS",
    )

    # Step 5: Display results
    print("\n" + "=" * 70)
    print("A) CRYPTO-LED DRAWDOWN EPISODES")
    print("=" * 70)
    if "A_crypto_led_drawdowns" in results and not results["A_crypto_led_drawdowns"].empty:
        print(results["A_crypto_led_drawdowns"].to_string(index=False))
    else:
        print("No significant crypto-led drawdown episodes detected.")

    print("\n" + "=" * 70)
    print("B) EQUITY MACRO STRESS EPISODES")
    print("=" * 70)
    if "B_equity_macro_stress" in results and not results["B_equity_macro_stress"].empty:
        print(results["B_equity_macro_stress"].to_string(index=False))
    else:
        print("No significant equity macro stress episodes detected.")

    print("\n" + "=" * 70)
    print("C) CORRELATION REGIMES (PORT vs BENCH)")
    print("=" * 70)
    if "C_correlation_regimes" in results and not results["C_correlation_regimes"].empty:
        print(results["C_correlation_regimes"].to_string(index=False))
    else:
        print("No correlation regime data available.")

    print("\n" + "=" * 70)
    print("D) VOLATILITY & DRAWDOWN ENVELOPE (last 15 rows)")
    print("=" * 70)
    if "D_vol_dd_envelope" in results and not results["D_vol_dd_envelope"].empty:
        print(results["D_vol_dd_envelope"].tail(15).to_string())
    else:
        print("No envelope data available.")

    print("\n✅ Financial stress tests complete!")
    print("📁 Results saved to: STRESS_A_*.csv, STRESS_B_*.csv, STRESS_C_*.csv, STRESS_D_*.csv")
