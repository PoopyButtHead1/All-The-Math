# stress_testing.py
# Regime stress testing for ONCH-style backtests:
# 1) Crypto-led drawdowns
# 2) Equity macro stress
# 3) Correlation regime shifts
# 4) Volatility & drawdown envelope
#
# Works with the result dict returned by run_backtest() in your ONCH_Backtest.py.
# Optionally pulls additional proxy series via Polygon if you want (BTC proxy, SPY, etc.)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Core helpers
# ----------------------------

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.sort_index()
    raise TypeError("Expected pandas Series")

def nav_from_returns(daily_ret: pd.Series, start_nav: float = 1.0) -> pd.Series:
    r = _to_series(daily_ret).dropna()
    nav = (1.0 + r).cumprod() * float(start_nav)
    nav.name = "NAV"
    return nav

def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = _to_series(nav).dropna()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    dd.name = "Drawdown"
    return dd

def annualize_vol(daily_ret: pd.Series, periods_per_year: int = 252) -> float:
    r = _to_series(daily_ret).dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))

def annualize_cagr(daily_ret: pd.Series, periods_per_year: int = 252) -> float:
    r = _to_series(daily_ret).dropna()
    n = len(r)
    if n == 0:
        return np.nan
    total = float((1.0 + r).prod())
    return total ** (periods_per_year / n) - 1.0

def max_drawdown(nav: pd.Series) -> float:
    dd = drawdown_series(nav)
    return float(dd.min()) if len(dd) else np.nan

def worst_drawdown_window(dd: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp, float]:
    """
    Returns (peak_date, trough_date, min_dd) for the worst drawdown episode.
    """
    dd = _to_series(dd).dropna()
    if dd.empty:
        return (pd.NaT, pd.NaT, np.nan)
    trough_date = dd.idxmin()
    min_dd = float(dd.loc[trough_date])

    # Peak is the last date before trough where dd == 0
    pre = dd.loc[:trough_date]
    zeros = pre[pre >= -1e-12]
    if len(zeros) == 0:
        peak_date = dd.index.min()
    else:
        peak_date = zeros.index.max()
    return peak_date, trough_date, min_dd

def find_drawdown_episodes(
    nav: pd.Series,
    threshold: float = -0.20,
    min_days: int = 10,
    include_recovery: bool = True,
) -> List[Dict]:
    """
    Detect drawdown episodes where drawdown breaches `threshold` (e.g. -0.30),
    and returns list of episodes with dates + severity.
    """
    nav = _to_series(nav).dropna()
    dd = drawdown_series(nav)

    episodes = []
    in_episode = False
    start = None
    trough_date = None
    trough_dd = 0.0

    for d, v in dd.items():
        if (not in_episode) and (v <= threshold):
            # start episode at prior peak (most recent dd==0)
            pre = dd.loc[:d]
            zeros = pre[pre >= -1e-12]
            start = zeros.index.max() if len(zeros) else dd.index.min()
            in_episode = True
            trough_date = d
            trough_dd = float(v)

        if in_episode:
            if v < trough_dd:
                trough_dd = float(v)
                trough_date = d

            # end episode when recovered (dd back to ~0)
            if include_recovery and (v >= -1e-12):
                end = d
                duration = (dd.loc[start:end].shape[0])  # trading days
                if duration >= min_days:
                    episodes.append(
                        {
                            "Start": start,
                            "Trough": trough_date,
                            "End": end,
                            "MinDrawdown": trough_dd,
                            "DurationDays": int(duration),
                        }
                    )
                in_episode = False
                start = None
                trough_date = None
                trough_dd = 0.0

    # If episode never recovers in sample, close at last date
    if in_episode:
        end = dd.index.max()
        duration = (dd.loc[start:end].shape[0])
        if duration >= min_days:
            episodes.append(
                {
                    "Start": start,
                    "Trough": trough_date,
                    "End": end,
                    "MinDrawdown": trough_dd,
                    "DurationDays": int(duration),
                }
            )

    # Sort worst first
    episodes = sorted(episodes, key=lambda x: x["MinDrawdown"])
    return episodes

def slice_metrics(
    port_ret: pd.Series,
    bench_ret: Optional[pd.Series] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    port_ret = _to_series(port_ret)
    if start is not None or end is not None:
        port_ret = port_ret.loc[start:end]
    port_ret = port_ret.dropna()

    out = {
        "Days": int(len(port_ret)),
        "TotalReturn": float((1.0 + port_ret).prod() - 1.0) if len(port_ret) else np.nan,
        "CAGR": annualize_cagr(port_ret, periods_per_year),
        "Vol": annualize_vol(port_ret, periods_per_year),
        "MaxDD": max_drawdown(nav_from_returns(port_ret)),
    }

    if bench_ret is not None:
        bench_ret = _to_series(bench_ret)
        if start is not None or end is not None:
            bench_ret = bench_ret.loc[start:end]
        df = pd.concat([port_ret.rename("p"), bench_ret.rename("b")], axis=1).dropna()
        if len(df):
            out["ActiveTotal"] = float((1.0 + df["p"]).prod() - (1.0 + df["b"]).prod())
            out["HitRateVsBench"] = float((df["p"] > df["b"]).mean())
            out["Corr"] = float(df["p"].corr(df["b"]))
        else:
            out["ActiveTotal"] = np.nan
            out["HitRateVsBench"] = np.nan
            out["Corr"] = np.nan

    return out


# ----------------------------
# Correlation regime logic
# ----------------------------

@dataclass
class CorrRegimeConfig:
    window: int = 63          # ~3 months
    high_corr_q: float = 0.80 # top 20% is "high corr"
    low_corr_q: float = 0.20  # bottom 20% is "low corr"

def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([_to_series(a).rename("a"), _to_series(b).rename("b")], axis=1).dropna()
    c = df["a"].rolling(window).corr(df["b"])
    c.name = "RollCorr"
    return c

def label_corr_regimes(corr: pd.Series, cfg: CorrRegimeConfig) -> pd.Series:
    corr = _to_series(corr).dropna()
    if corr.empty:
        return pd.Series(dtype="object")

    hi = float(corr.quantile(cfg.high_corr_q))
    lo = float(corr.quantile(cfg.low_corr_q))

    lab = pd.Series(index=corr.index, dtype="object")
    lab[corr >= hi] = "HighCorr"
    lab[corr <= lo] = "LowCorr"
    lab[(corr > lo) & (corr < hi)] = "MidCorr"
    lab.name = "CorrRegime"
    return lab

def regime_summary_returns(
    port_ret: pd.Series,
    regime_labels: pd.Series,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    df = pd.concat([_to_series(port_ret).rename("r"), _to_series(regime_labels).rename("reg")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for reg, sub in df.groupby("reg"):
        r = sub["r"]
        rows.append(
            {
                "Regime": reg,
                "Days": int(len(r)),
                "TotalReturn": float((1.0 + r).prod() - 1.0),
                "CAGR": annualize_cagr(r, periods_per_year),
                "Vol": annualize_vol(r, periods_per_year),
                "MaxDD": max_drawdown(nav_from_returns(r)),
            }
        )
    return pd.DataFrame(rows).sort_values("Regime")


# ----------------------------
# Stress test suites
# ----------------------------

def crypto_led_drawdowns_suite(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    crypto_proxy_ret: pd.Series,
    dd_threshold: float = -0.30,
    min_days: int = 10,
    periods_per_year: int = 252,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Defines 'crypto-led drawdown' episodes as episodes where the CRYPTO PROXY NAV
    experiences drawdowns <= dd_threshold, then reports ONCH metrics over the same windows.
    """
    crypto_nav = nav_from_returns(crypto_proxy_ret)
    episodes = find_drawdown_episodes(crypto_nav, threshold=dd_threshold, min_days=min_days, include_recovery=True)

    rows = []
    for ep in episodes[:top_n]:
        start, end = ep["Start"], ep["End"]
        m_port = slice_metrics(port_ret, bench_ret, start, end, periods_per_year)
        m_crypto = slice_metrics(crypto_proxy_ret, None, start, end, periods_per_year)

        rows.append(
            {
                "EpisodeStart": start,
                "EpisodeEnd": end,
                "CryptoMinDD": ep["MinDrawdown"],
                "CryptoTotal": m_crypto["TotalReturn"],
                "PortTotal": m_port["TotalReturn"],
                "BenchTotal": (slice_metrics(bench_ret, None, start, end, periods_per_year))["TotalReturn"],
                "PortMaxDD": m_port["MaxDD"],
                "PortVol": m_port["Vol"],
                "PortCorrToBench": m_port.get("Corr", np.nan),
                "HitRateVsBench": m_port.get("HitRateVsBench", np.nan),
                "Days": m_port["Days"],
            }
        )

    return pd.DataFrame(rows)

def equity_macro_stress_suite(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    equity_proxy_ret: pd.Series,
    dd_threshold: float = -0.15,
    min_days: int = 10,
    periods_per_year: int = 252,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Defines 'equity macro stress' episodes as episodes where the EQUITY PROXY NAV
    experiences drawdowns <= dd_threshold (e.g. -15%).
    """
    eq_nav = nav_from_returns(equity_proxy_ret)
    episodes = find_drawdown_episodes(eq_nav, threshold=dd_threshold, min_days=min_days, include_recovery=True)

    rows = []
    for ep in episodes[:top_n]:
        start, end = ep["Start"], ep["End"]
        m_port = slice_metrics(port_ret, bench_ret, start, end, periods_per_year)
        m_eq = slice_metrics(equity_proxy_ret, None, start, end, periods_per_year)
        m_bench = slice_metrics(bench_ret, None, start, end, periods_per_year)

        rows.append(
            {
                "EpisodeStart": start,
                "EpisodeEnd": end,
                "EquityMinDD": ep["MinDrawdown"],
                "EquityTotal": m_eq["TotalReturn"],
                "PortTotal": m_port["TotalReturn"],
                "BenchTotal": m_bench["TotalReturn"],
                "PortMaxDD": m_port["MaxDD"],
                "PortVol": m_port["Vol"],
                "PortCorrToBench": m_port.get("Corr", np.nan),
                "HitRateVsBench": m_port.get("HitRateVsBench", np.nan),
                "Days": m_port["Days"],
            }
        )

    return pd.DataFrame(rows)

def correlation_regime_shifts_suite(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    crypto_proxy_ret: Optional[pd.Series] = None,
    cfg: CorrRegimeConfig = CorrRegimeConfig(),
    periods_per_year: int = 252,
) -> Dict[str, pd.DataFrame]:
    """
    Produces regime labels based on rolling correlation and summarizes performance by regime.
    - Regime 1: Corr(port, bench) high/mid/low
    - Optional: Corr(crypto, bench) high/mid/low + Corr(port, crypto)
    """
    out = {}

    c_pb = rolling_corr(port_ret, bench_ret, cfg.window)
    lab_pb = label_corr_regimes(c_pb, cfg)
    out["Port_vs_Bench_RegimeSummary"] = regime_summary_returns(port_ret, lab_pb, periods_per_year)

    if crypto_proxy_ret is not None:
        c_cb = rolling_corr(crypto_proxy_ret, bench_ret, cfg.window)
        lab_cb = label_corr_regimes(c_cb, cfg)
        out["Crypto_vs_Bench_RegimeSummary"] = regime_summary_returns(port_ret, lab_cb, periods_per_year)

        c_pc = rolling_corr(port_ret, crypto_proxy_ret, cfg.window)
        lab_pc = label_corr_regimes(c_pc, cfg)
        out["Port_vs_Crypto_RegimeSummary"] = regime_summary_returns(port_ret, lab_pc, periods_per_year)

    return out

def volatility_and_drawdown_envelope_suite(
    port_ret: pd.Series,
    bench_ret: Optional[pd.Series] = None,
    window: int = 252,  # 12m
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Rolling stats:
    - rolling vol
    - rolling max drawdown (computed from rolling NAVs)
    - rolling total return
    Also returns bench equivalents if provided.
    """
    r = _to_series(port_ret).dropna()
    nav = nav_from_returns(r)

    roll_vol = r.rolling(window).std(ddof=1) * math.sqrt(periods_per_year)
    roll_total = (nav / nav.shift(window)) - 1.0

    # rolling max drawdown: compute drawdown on rolling window NAV path
    roll_mdd = pd.Series(index=r.index, dtype=float)
    for i in range(window, len(nav)):
        wnav = nav.iloc[i - window : i + 1]
        roll_mdd.iloc[i] = max_drawdown(wnav)

    df = pd.DataFrame(
        {
            "Port_RollVol": roll_vol,
            "Port_RollTotal": roll_total,
            "Port_RollMaxDD": roll_mdd,
        }
    )

    if bench_ret is not None:
        b = _to_series(bench_ret).reindex(df.index).dropna()
        bnav = nav_from_returns(b)
        b_roll_vol = b.rolling(window).std(ddof=1) * math.sqrt(periods_per_year)
        b_roll_total = (bnav / bnav.shift(window)) - 1.0

        b_roll_mdd = pd.Series(index=b.index, dtype=float)
        for i in range(window, len(bnav)):
            wnav = bnav.iloc[i - window : i + 1]
            b_roll_mdd.iloc[i] = max_drawdown(wnav)

        df["Bench_RollVol"] = b_roll_vol
        df["Bench_RollTotal"] = b_roll_total
        df["Bench_RollMaxDD"] = b_roll_mdd

    return df


# ----------------------------
# Plotting helpers (matplotlib; no fixed colors)
# ----------------------------

def plot_underwater(nav: pd.Series, title: str = "Underwater (Drawdown)") -> None:
    nav = _to_series(nav)
    dd = drawdown_series(nav)
    plt.figure()
    plt.plot(dd.index, dd.values)
    plt.axhline(0.0)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

def plot_rolling_envelope(envelope_df: pd.DataFrame, title_prefix: str = "Rolling 12M") -> None:
    df = envelope_df.dropna(how="all")
    if df.empty:
        print("No envelope data to plot (insufficient history for rolling window).")
        return

    if "Port_RollVol" in df:
        plt.figure()
        cols = [c for c in df.columns if c.endswith("RollVol")]
        for c in cols:
            plt.plot(df.index, df[c].values, label=c)
        plt.title(f"{title_prefix} Volatility")
        plt.xlabel("Date")
        plt.ylabel("Ann. Vol")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if "Port_RollMaxDD" in df:
        plt.figure()
        cols = [c for c in df.columns if c.endswith("RollMaxDD")]
        for c in cols:
            plt.plot(df.index, df[c].values, label=c)
        plt.title(f"{title_prefix} Max Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Max DD (window)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if "Port_RollTotal" in df:
        plt.figure()
        cols = [c for c in df.columns if c.endswith("RollTotal")]
        for c in cols:
            plt.plot(df.index, df[c].values, label=c)
        plt.title(f"{title_prefix} Total Return")
        plt.xlabel("Date")
        plt.ylabel("Total return (window)")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_rolling_correlations(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    crypto_proxy_ret: Optional[pd.Series] = None,
    window: int = 63,
) -> None:
    c_pb = rolling_corr(port_ret, bench_ret, window)
    plt.figure()
    plt.plot(c_pb.index, c_pb.values, label="Corr(Port, Bench)")
    if crypto_proxy_ret is not None:
        c_cb = rolling_corr(crypto_proxy_ret, bench_ret, window)
        c_pc = rolling_corr(port_ret, crypto_proxy_ret, window)
        plt.plot(c_cb.index, c_cb.values, label="Corr(Crypto, Bench)")
        plt.plot(c_pc.index, c_pc.values, label="Corr(Port, Crypto)")
    plt.axhline(0.0)
    plt.title(f"Rolling Correlations ({window}d)")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# One-shot runner
# ----------------------------

@dataclass
class StressConfig:
    periods_per_year: int = 252

    # episode thresholds
    crypto_dd_threshold: float = -0.30
    equity_dd_threshold: float = -0.15
    min_episode_days: int = 10
    top_n_episodes: int = 5

    # correlation regime
    corr_window: int = 63
    corr_hi_q: float = 0.80
    corr_lo_q: float = 0.20

    # envelope
    envelope_window: int = 252

def run_all_stress_tests(
    result: Dict[str, object],
    crypto_proxy_ret: Optional[pd.Series] = None,
    equity_proxy_ret: Optional[pd.Series] = None,
    cfg: StressConfig = StressConfig(),
    make_plots: bool = True,
) -> Dict[str, object]:
    """
    `result` is the dict from run_backtest():
      result["port_ret"], result["bench_ret"], result["nav"], result["bench_nav"], etc.

    Provide:
      - crypto_proxy_ret (Series) for crypto-led drawdowns & correlation regimes
      - equity_proxy_ret (Series) for equity macro stress (SPY, QQQ, etc.)
        (If you omit equity_proxy_ret, it will use benchmark returns as the equity proxy.)
    """
    port_ret = _to_series(result["port_ret"])
    bench_ret = _to_series(result["bench_ret"])
    nav = _to_series(result["nav"])

    if equity_proxy_ret is None:
        equity_proxy_ret = bench_ret

    outputs: Dict[str, object] = {}

    # 1) Crypto-led drawdowns
    if crypto_proxy_ret is not None:
        crypto_table = crypto_led_drawdowns_suite(
            port_ret=port_ret,
            bench_ret=bench_ret,
            crypto_proxy_ret=_to_series(crypto_proxy_ret),
            dd_threshold=cfg.crypto_dd_threshold,
            min_days=cfg.min_episode_days,
            periods_per_year=cfg.periods_per_year,
            top_n=cfg.top_n_episodes,
        )
        outputs["crypto_led_drawdowns"] = crypto_table

    # 2) Equity macro stress
    eq_table = equity_macro_stress_suite(
        port_ret=port_ret,
        bench_ret=bench_ret,
        equity_proxy_ret=_to_series(equity_proxy_ret),
        dd_threshold=cfg.equity_dd_threshold,
        min_days=cfg.min_episode_days,
        periods_per_year=cfg.periods_per_year,
        top_n=cfg.top_n_episodes,
    )
    outputs["equity_macro_stress"] = eq_table

    # 3) Correlation regime shifts
    corr_cfg = CorrRegimeConfig(
        window=cfg.corr_window,
        high_corr_q=cfg.corr_hi_q,
        low_corr_q=cfg.corr_lo_q,
    )
    outputs["correlation_regimes"] = correlation_regime_shifts_suite(
        port_ret=port_ret,
        bench_ret=bench_ret,
        crypto_proxy_ret=_to_series(crypto_proxy_ret) if crypto_proxy_ret is not None else None,
        cfg=corr_cfg,
        periods_per_year=cfg.periods_per_year,
    )

    # 4) Volatility & drawdown envelope
    envelope = volatility_and_drawdown_envelope_suite(
        port_ret=port_ret,
        bench_ret=bench_ret,
        window=cfg.envelope_window,
        periods_per_year=cfg.periods_per_year,
    )
    outputs["vol_drawdown_envelope"] = envelope

    if make_plots:
        plot_underwater(nav, title="ONCH Underwater (Drawdown)")
        plot_rolling_envelope(envelope, title_prefix=f"Rolling {cfg.envelope_window}D")
        if crypto_proxy_ret is not None:
            plot_rolling_correlations(port_ret, bench_ret, crypto_proxy_ret=_to_series(crypto_proxy_ret), window=cfg.corr_window)
        else:
            plot_rolling_correlations(port_ret, bench_ret, crypto_proxy_ret=None, window=cfg.corr_window)

    return outputs


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    """
    Integration with your ONCH_Backtest.py:
    This script runs stress tests on the backtest results.
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
    print("REGIME STRESS TEST SUITE")
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

    # Step 3: Download proxies for stress testing
    print("\nDownloading crypto proxy (COIN - for crypto-led stress tests)...")
    time.sleep(5)
    coin_px = polygon_get_adjusted_closes(
        ticker="COIN",
        start=START,
        end=END,
        api_key=POLYGON_API_KEY,
        adjusted=True,
    )
    crypto_proxy_ret = coin_px.pct_change().dropna()

    print("Downloading equity proxy (SPY - for macro stress tests)...")
    time.sleep(5)
    spy_px = polygon_get_adjusted_closes(
        ticker="SPY",
        start=START,
        end=END,
        api_key=POLYGON_API_KEY,
        adjusted=True,
    )
    equity_proxy_ret = spy_px.pct_change().dropna()

    # Step 4: Run stress tests
    print("\n" + "=" * 70)
    print("Running stress tests...")
    print("=" * 70)

    cfg_stress = StressConfig(
        periods_per_year=252,
        crypto_dd_threshold=-0.30,
        equity_dd_threshold=-0.15,
        min_episode_days=10,
        top_n_episodes=5,
        corr_window=63,
        corr_hi_q=0.80,
        corr_lo_q=0.20,
        envelope_window=252,
    )

    outputs = run_all_stress_tests(
        result,
        crypto_proxy_ret=crypto_proxy_ret,
        equity_proxy_ret=equity_proxy_ret,
        cfg=cfg_stress,
        make_plots=True,
    )

    # Step 5: Display results
    print("\n" + "=" * 70)
    print("CRYPTO-LED DRAWDOWNS")
    print("=" * 70)
    if "crypto_led_drawdowns" in outputs:
        print(outputs["crypto_led_drawdowns"].to_string())

    print("\n" + "=" * 70)
    print("EQUITY MACRO STRESS")
    print("=" * 70)
    print(outputs["equity_macro_stress"].to_string())

    print("\n" + "=" * 70)
    print("CORRELATION REGIME: PORT vs BENCH")
    print("=" * 70)
    regimes = outputs["correlation_regimes"]
    print(regimes["Port_vs_Bench_RegimeSummary"].to_string())

    print("\n" + "=" * 70)
    print("VOLATILITY & DRAWDOWN ENVELOPE (last 20 rows)")
    print("=" * 70)
    print(outputs["vol_drawdown_envelope"].tail(20).to_string())

    print("\nStress tests complete!")
