# rules_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class RuleEngineConfig:
    vol_window: int = 63
    dd_window: int = 252
    corr_window: int = 63

    # triggers
    vol_mult_limit: float = 1.4
    drawdown_trigger: float = -0.25
    high_corr_trigger: float = 0.80

    # actions
    pillar_a_boost_on_breach: float = 0.10
    stress_cap_overrides: Optional[Dict[str, float]] = None


def _ann_vol(ret: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
    """Compute rolling annualized volatility."""
    return ret.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def _drawdown(nav: pd.Series) -> pd.Series:
    """Compute drawdown series from NAV."""
    peak = nav.cummax()
    return nav / peak - 1.0


def _align_series(pr: pd.Series, br: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Align two series to common dates and drop NaNs."""
    idx = pr.index.intersection(br.index)
    pr = pr.reindex(idx).dropna()
    br = br.reindex(idx).dropna()
    
    # re-intersect after dropping NaNs
    idx = pr.index.intersection(br.index)
    return pr.reindex(idx), br.reindex(idx)


def compute_signals(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    cfg: RuleEngineConfig = RuleEngineConfig(),
) -> pd.DataFrame:
    """
    Compute all trading signals: vol breach, drawdown trigger, high correlation.
    Returns DataFrame with signals indexed by date.
    """
    pr, br = _align_series(port_ret, bench_ret)
    
    if pr.empty or br.empty:
        raise ValueError("No overlapping dates between portfolio and benchmark returns.")

    port_nav = (1.0 + pr).cumprod()
    bench_nav = (1.0 + br).cumprod()

    sig = pd.DataFrame(index=pr.index)
    sig["PortVol"] = _ann_vol(pr, cfg.vol_window)
    sig["BenchVol"] = _ann_vol(br, cfg.vol_window)
    sig["VolBreach"] = sig["PortVol"] > (cfg.vol_mult_limit * sig["BenchVol"])

    sig["PortDD"] = _drawdown(port_nav)
    sig["DDTrigger"] = sig["PortDD"] <= cfg.drawdown_trigger

    sig["RollCorr"] = pr.rolling(cfg.corr_window).corr(br)
    sig["HighCorr"] = sig["RollCorr"] >= cfg.high_corr_trigger

    return sig.dropna()


def _clip_to_ranges(
    targets: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    """Clip targets to min/max ranges and renormalize to 1.0."""
    out = targets.copy()
    
    # Clip each pillar
    for pillar, (lo, hi) in ranges.items():
        if pillar in out:
            out[pillar] = float(np.clip(out[pillar], lo, hi))
    
    # Renormalize to 1.0
    total = sum(out.values())
    if total <= 0:
        raise ValueError(f"Pillar targets sum to {total} after clipping. Invalid ranges.")
    
    return {k: v / total for k, v in out.items()}


def apply_rules_on_rebalance(
    asof_date: pd.Timestamp,
    signals: pd.DataFrame,
    base_targets: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
    cfg: RuleEngineConfig = RuleEngineConfig(),
) -> Dict[str, object]:
    """
    Apply trading rules based on latest signals and return adjusted targets.
    
    Returns dict with:
      - pillar_targets: adjusted allocation targets
      - cap_overrides: stress caps (if rules fired)
      - fired_rules: list of rule names that triggered
      - signals_row: the signal values used
    """
    # Get signal row as of date (use last available if exact date missing)
    if asof_date not in signals.index:
        srow = signals.loc[:asof_date].tail(1)
        if srow.empty:
            return {
                "pillar_targets": base_targets,
                "cap_overrides": None,
                "fired_rules": [],
                "signals_row": {}
            }
        row = srow.iloc[0]
    else:
        row = signals.loc[asof_date]

    targets = base_targets.copy()
    fired = []

    # Rule 1: Drawdown Trigger => Boost Pillar A (defensive)
    if pd.notna(row.get("DDTrigger")) and bool(row["DDTrigger"]):
        fired.append("DDTrigger")
        a_lo, a_hi = ranges["A"]
        boost = min(targets["A"] + cfg.pillar_a_boost_on_breach, a_hi)
        targets["A"] = boost

    # Rule 2: Volatility Breach => Boost Pillar A
    if pd.notna(row.get("VolBreach")) and bool(row["VolBreach"]):
        fired.append("VolBreach")
        a_lo, a_hi = ranges["A"]
        boost = min(targets["A"] + cfg.pillar_a_boost_on_breach, a_hi)
        targets["A"] = boost

    # Rule 3: High Correlation => Defensive tilt (A + D)
    if pd.notna(row.get("HighCorr")) and bool(row["HighCorr"]):
        fired.append("HighCorr")
        a_lo, a_hi = ranges["A"]
        d_lo, d_hi = ranges["D"]
        targets["A"] = min(targets["A"] + 0.02, a_hi)
        targets["D"] = min(targets["D"] + 0.02, d_hi)

    # Normalize and clip
    total = sum(targets.values())
    if total > 0:
        targets = {k: v / total for k, v in targets.items()}

    targets = _clip_to_ranges(targets, ranges)

    cap_overrides = cfg.stress_cap_overrides if fired else None

    return {
        "pillar_targets": targets,
        "cap_overrides": cap_overrides,
        "fired_rules": fired,
        "signals_row": row.to_dict(),
    }


def backtest_with_rules(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    base_targets: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
    rebalance_freq: str = "ME",
    cfg: RuleEngineConfig = RuleEngineConfig(),
) -> pd.DataFrame:
    """
    Run full backtest applying rules-based rebalancing.
    
    Returns DataFrame with:
      - dates, portfolio returns, applied targets, fired rules
    """
    signals = compute_signals(port_ret, bench_ret, cfg)
    
    # Get rebalance dates
    pr, br = _align_series(port_ret, bench_ret)
    rebalance_dates = pr.asfreq(rebalance_freq).index
    
    rows = []
    for rebal_date in rebalance_dates:
        if rebal_date not in pr.index:
            continue
        
        result = apply_rules_on_rebalance(rebal_date, signals, base_targets, ranges, cfg)
        
        rows.append({
            "Date": rebal_date,
            "PortfolioReturn": pr.get(rebal_date, np.nan),
            "BenchmarkReturn": br.get(rebal_date, np.nan),
            "A_Target": result["pillar_targets"].get("A", np.nan),
            "B_Target": result["pillar_targets"].get("B", np.nan),
            "C_Target": result["pillar_targets"].get("C", np.nan),
            "D_Target": result["pillar_targets"].get("D", np.nan),
            "FiredRules": "|".join(result["fired_rules"]),
            "VolBreach": result["signals_row"].get("VolBreach", False),
            "DDTrigger": result["signals_row"].get("DDTrigger", False),
            "HighCorr": result["signals_row"].get("HighCorr", False),
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    """
    Example: Load backtest results and run rules engine.
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

    START = "2018-01-01"
    END = "2024-12-31"

    print("=" * 70)
    print("RULES ENGINE BACKTEST")
    print("=" * 70)

    # Step 1: Download and run backtest
    WEIGHTS = normalize_weights(ONCH_WEIGHTS)
    TICKERS = list(WEIGHTS.keys())
    BENCH = DEFAULT_BENCHMARK

    print("\nDownloading portfolio data...")
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

    print("\nRunning backtest...")
    available = [t for t in TICKERS if t in prices.columns and prices[t].notna().any()]
    WEIGHTS = normalize_weights({t: WEIGHTS[t] for t in available})

    cfg_bt = BacktestConfig(
        weights=WEIGHTS,
        benchmark=BENCH,
        trading_cost_bps=10.0,
        mgmt_fee_bps_annual=0.0,
        rf_annual=0.0,
        require_full_history=True,
    )

    result = run_backtest(prices, bench_prices, cfg_bt)

    # Step 2: Define pillar allocation ranges
    base_targets = {"A": 0.20, "B": 0.30, "C": 0.30, "D": 0.20}
    ranges = {
        "A": (0.15, 0.35),
        "B": (0.20, 0.40),
        "C": (0.20, 0.40),
        "D": (0.15, 0.30),
    }

    cfg_rules = RuleEngineConfig(
        vol_mult_limit=1.4,
        drawdown_trigger=-0.25,
        high_corr_trigger=0.80,
        pillar_a_boost_on_breach=0.10,
    )

    # Step 3: Compute signals
    print("\nComputing trading signals...")
    signals = compute_signals(result["port_ret"], result["bench_ret"], cfg_rules)
    print(f"✅ Computed {len(signals)} signal rows")
    print("\nSignal summary (first 10):")
    print(signals.head(10).to_string())

    # Step 4: Run backtest with rules
    print("\n" + "=" * 70)
    print("Running rules-based rebalancing...")
    print("=" * 70)
    results = backtest_with_rules(
        result["port_ret"],
        result["bench_ret"],
        base_targets,
        ranges,
        rebalance_freq="ME",
        cfg=cfg_rules,
    )
    print(f"✅ Rules-based backtest: {len(results)} rebalance dates")
    print("\nRebalancing activity (first 15):")
    print(results.head(15).to_string(index=False))

    # Step 5: Save results
    results.to_csv("RulesEngine_Results.csv", index=False)
    signals.to_csv("RulesEngine_Signals.csv")
    print("\n✅ Results saved:")
    print("   - RulesEngine_Results.csv")
    print("   - RulesEngine_Signals.csv")
