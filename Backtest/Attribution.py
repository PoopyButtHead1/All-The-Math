# attribution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class AttributionConfig:
    # if you want “pillars by month” set this to "M", else keep daily
    resample: Optional[str] = None  # e.g. "M"


def compute_contributions(weights_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    contributions[ticker] = weight[ticker] * return[ticker]
    weights_df: index dates, columns tickers
    returns_df: index dates, columns tickers
    """
    w = weights_df.copy().sort_index()
    r = returns_df.copy().sort_index()
    common = w.index.intersection(r.index)
    w = w.reindex(common).fillna(method="ffill").fillna(0.0)
    r = r.reindex(common).fillna(0.0)
    contrib = w * r
    return contrib


def aggregate_by_group(contrib_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Sum contributions into groups (pillars).
    mapping: ticker -> group
    """
    groups = sorted(set(mapping.values()))
    out = pd.DataFrame(index=contrib_df.index, columns=groups, dtype=float)
    out.loc[:, :] = 0.0

    for t in contrib_df.columns:
        g = mapping.get(t, "UNMAPPED")
        if g not in out.columns:
            out[g] = 0.0
        out[g] = out[g] + contrib_df[t].fillna(0.0)

    return out


def label_regimes_daily(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    crypto_ret: Optional[pd.Series] = None,
    corr_window: int = 63,
    vol_window: int = 63,
    high_corr_q: float = 0.80,
    low_corr_q: float = 0.20,
    equity_dd_trigger: float = -0.10,
    crypto_dd_trigger: float = -0.30,
) -> pd.Series:
    """
    Produces ONE daily regime label per day:
    Priority:
      1) CryptoStress (if crypto drawdown <= crypto_dd_trigger)
      2) EquityStress (if bench drawdown <= equity_dd_trigger)
      3) Corr regime (HighCorr / LowCorr / MidCorr based on rolling corr quantiles)
    """
    pr = port_ret.dropna().sort_index()
    br = bench_ret.dropna().sort_index()
    common = pr.index.intersection(br.index)
    pr = pr.reindex(common)
    br = br.reindex(common)

    # rolling corr
    rc = pr.rolling(corr_window).corr(br).dropna()
    hi = float(rc.quantile(high_corr_q)) if len(rc) else np.nan
    lo = float(rc.quantile(low_corr_q)) if len(rc) else np.nan

    corr_reg = pd.Series(index=pr.index, dtype="object")
    corr_reg.loc[:] = "MidCorr"
    corr_reg.loc[rc.index[rc >= hi]] = "HighCorr"
    corr_reg.loc[rc.index[rc <= lo]] = "LowCorr"

    # bench drawdown
    bench_nav = (1.0 + br.fillna(0.0)).cumprod()
    bench_dd = bench_nav / bench_nav.cummax() - 1.0

    regime = corr_reg.copy()
    regime[bench_dd <= equity_dd_trigger] = "EquityStress"

    if crypto_ret is not None:
        cr = crypto_ret.dropna().sort_index().reindex(pr.index).dropna()
        # align by intersection
        common2 = regime.index.intersection(cr.index)
        regime = regime.reindex(common2)
        pr = pr.reindex(common2)
        br = br.reindex(common2)
        cr = cr.reindex(common2)

        crypto_nav = (1.0 + cr.fillna(0.0)).cumprod()
        crypto_dd = crypto_nav / crypto_nav.cummax() - 1.0
        regime[crypto_dd <= crypto_dd_trigger] = "CryptoStress"

    regime.name = "Regime"
    return regime


def summarize_by_regime(
    contrib_groups: pd.DataFrame,
    regime: pd.Series,
    cfg: AttributionConfig = AttributionConfig(),
) -> pd.DataFrame:
    """
    Returns regime table with:
      - TotalReturn (sum of contributions) per regime
      - GroupShare: group contribution / total contribution in that regime
    """
    df = pd.concat([contrib_groups, regime], axis=1).dropna()
    if cfg.resample:
        # resample contributions then relabel regime by last label in period
        contrib = contrib_groups.resample(cfg.resample).sum()
        reg = regime.resample(cfg.resample).last()
        df = pd.concat([contrib, reg], axis=1).dropna()

    groups = [c for c in df.columns if c != "Regime"]

    rows = []
    for reg_name, sub in df.groupby("Regime"):
        gsum = sub[groups].sum()
        total = float(gsum.sum())
        row = {"Regime": reg_name, "TotalReturnApprox": total}
        for g in groups:
            row[f"{g}_Share"] = float(gsum[g] / total) if abs(total) > 1e-12 else np.nan
            row[f"{g}_Total"] = float(gsum[g])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Regime")
    return out
