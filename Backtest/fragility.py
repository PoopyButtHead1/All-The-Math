# fragility.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    name: str
    cap_overrides: Optional[Dict[str, float]] = None       # e.g. {"MSTR": 0.02}
    blacklist: Optional[List[str]] = None                  # e.g. ["GLXY"]
    shock_events: Optional[List[Tuple[pd.Timestamp, str, float]]] = None
    # shock event tuple: (date, ticker, shock_return) e.g. ("2022-06-15", "COIN", -0.50)


def apply_blacklist(weights: Dict[str, float], blacklist: List[str]) -> Dict[str, float]:
    w = {k: v for k, v in weights.items() if k not in set(blacklist)}
    s = sum(w.values())
    if s <= 0:
        raise ValueError("All holdings removed by blacklist.")
    return {k: v / s for k, v in w.items()}


def apply_caps(weights: Dict[str, float], caps: Dict[str, float]) -> Dict[str, float]:
    """
    Simple capping + redistribute excess pro-rata to uncapped names.
    """
    w = weights.copy()
    # first cap
    excess = 0.0
    uncapped = []
    for k in list(w.keys()):
        cap = caps.get(k, None)
        if cap is None:
            uncapped.append(k)
            continue
        if w[k] > cap:
            excess += (w[k] - cap)
            w[k] = cap
        else:
            uncapped.append(k)

    if excess <= 0:
        # normalize to 1
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    # redistribute excess pro-rata to uncapped
    uncapped_sum = sum(w[k] for k in uncapped)
    if uncapped_sum <= 0:
        # if nothing uncapped, just renormalize
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    for k in uncapped:
        w[k] += excess * (w[k] / uncapped_sum)

    # final normalize
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def apply_shocks(returns_df: pd.DataFrame, shocks: List[Tuple[pd.Timestamp, str, float]]) -> pd.DataFrame:
    r = returns_df.copy()
    for d, ticker, shock_ret in shocks:
        d = pd.Timestamp(d)
        if d in r.index and ticker in r.columns:
            r.loc[d, ticker] = float(shock_ret)
    return r


def run_static_weights_backtest(
    returns_df: pd.DataFrame,
    base_weights: Dict[str, float],
    trading_cost_bps: float = 0.0,
    rebalance_dates: Optional[List[pd.Timestamp]] = None,
) -> pd.Series:
    """
    Lightweight backtest for scenario sweeps:
    - constant weights, optional rebalance dates (resets to target)
    """
    r = returns_df.dropna(how="any").copy()
    w_target = pd.Series(base_weights).reindex(r.columns).fillna(0.0)
    w_target = w_target / w_target.sum()

    w = w_target.copy()
    nav = 1.0
    out = []

    last_w = w.copy()
    for d in r.index:
        if rebalance_dates is not None and d in set(rebalance_dates):
            w = w_target.copy()
            turnover = float((w - last_w).abs().sum() / 2.0)
            nav *= (1.0 - turnover * (trading_cost_bps / 10000.0))
            last_w = w.copy()

        nav *= (1.0 + float((w * r.loc[d]).sum()))
        out.append((d, nav))

    return pd.Series(dict(out), name="NAV")


def scenario_runner(
    returns_df: pd.DataFrame,
    base_weights: Dict[str, float],
    scenario: Scenario,
    rebalance_dates: Optional[List[pd.Timestamp]] = None,
    trading_cost_bps: float = 0.0,
) -> Dict[str, object]:
    w = base_weights.copy()

    if scenario.blacklist:
        w = apply_blacklist(w, scenario.blacklist)

    if scenario.cap_overrides:
        w = apply_caps(w, scenario.cap_overrides)

    r = returns_df.copy()
    if scenario.shock_events:
        r = apply_shocks(r, scenario.shock_events)

    nav = run_static_weights_backtest(r, w, trading_cost_bps=trading_cost_bps, rebalance_dates=rebalance_dates)
    ret = nav.pct_change().dropna()

    mdd = float((nav / nav.cummax() - 1.0).min())
    total = float(nav.iloc[-1] - 1.0)

    return {
        "name": scenario.name,
        "weights": w,
        "nav": nav,
        "ret": ret,
        "total_return": total,
        "max_drawdown": mdd,
    }
