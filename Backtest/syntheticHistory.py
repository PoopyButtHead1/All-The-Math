# synthetic_history.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FactorModelConfig:
    window: int = 252               # rolling fit window (1y)
    min_fit_obs: int = 126          # minimum observations to fit
    ridge_lambda: float = 1e-6      # tiny ridge to stabilize inversion


def align_df(port_ret: pd.Series, factor_ret: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([port_ret.rename("port"), factor_ret], axis=1).dropna()
    return df


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> Tuple[float, np.ndarray]:
    """
    Ridge regression: minimize ||y - a - Xb||^2 + lam||b||^2
    Returns alpha, beta
    """
    # add intercept column
    ones = np.ones((X.shape[0], 1))
    X1 = np.hstack([ones, X])  # [1, factors...]

    # ridge only on betas, not intercept
    I = np.eye(X1.shape[1])
    I[0, 0] = 0.0

    A = X1.T @ X1 + lam * I
    b = X1.T @ y
    coef = np.linalg.solve(A, b)
    alpha = float(coef[0])
    beta = coef[1:].astype(float)
    return alpha, beta


def fit_factor_model_rolling(
    port_ret: pd.Series,
    factor_ret: pd.DataFrame,
    cfg: FactorModelConfig = FactorModelConfig(),
) -> pd.DataFrame:
    """
    Fits rolling factor model: port = alpha + sum(beta_k * factor_k)
    Returns DataFrame with columns: alpha, beta_<factor>
    indexed by fit end date.
    """
    df = align_df(port_ret, factor_ret)
    if df.shape[0] < cfg.min_fit_obs:
        raise ValueError("Not enough overlap between portfolio returns and factor returns to fit model.")

    params = []
    idx = df.index

    factors = [c for c in df.columns if c != "port"]
    X_all = df[factors].values
    y_all = df["port"].values

    for i in range(len(df)):
        start = max(0, i - cfg.window + 1)
        subX = X_all[start : i + 1, :]
        suby = y_all[start : i + 1]

        if subX.shape[0] < cfg.min_fit_obs:
            continue

        alpha, beta = _fit_ridge(subX, suby, cfg.ridge_lambda)
        row = {"alpha": alpha}
        for k, f in enumerate(factors):
            row[f"beta_{f}"] = float(beta[k])
        params.append((idx[i], row))

    out = pd.DataFrame([r for _, r in params], index=[d for d, _ in params]).sort_index()
    return out


def make_synthetic_returns(
    factor_ret_pre: pd.DataFrame,
    params_at_cutover: pd.Series,
) -> pd.Series:
    """
    Uses a single parameter set (alpha + betas) to generate synthetic returns for pre-cutover dates.
    """
    factor_ret_pre = factor_ret_pre.dropna(how="any").copy()
    factors = [c for c in factor_ret_pre.columns]
    alpha = float(params_at_cutover.get("alpha", 0.0))

    betas = np.array([float(params_at_cutover.get(f"beta_{f}", 0.0)) for f in factors], dtype=float)
    X = factor_ret_pre[factors].values
    y = alpha + X @ betas

    s = pd.Series(y, index=factor_ret_pre.index, name="port_synth")
    return s


def splice_returns(
    pre_ret: pd.Series,
    post_ret: pd.Series,
    cutover: pd.Timestamp,
) -> pd.Series:
    """
    pre_ret used for dates < cutover
    post_ret used for dates >= cutover
    """
    pre = pre_ret.loc[pre_ret.index < cutover]
    post = post_ret.loc[post_ret.index >= cutover]
    out = pd.concat([pre, post]).sort_index()
    return out


def build_long_horizon_portfolio_returns(
    port_ret_actual: pd.Series,
    factor_ret: pd.DataFrame,
    cutover: Optional[pd.Timestamp] = None,
    cfg: FactorModelConfig = FactorModelConfig(),
) -> Dict[str, object]:
    """
    1) Fit rolling factor model on actual overlap
    2) Choose params at cutover (or earliest actual date)
    3) Generate synthetic returns for dates before cutover
    4) Splice synthetic + actual
    """
    port_ret_actual = port_ret_actual.dropna().sort_index()
    factor_ret = factor_ret.dropna(how="any").sort_index()

    if cutover is None:
        cutover = port_ret_actual.index.min()

    params_df = fit_factor_model_rolling(port_ret_actual, factor_ret, cfg=cfg)

    # Use params as-of cutover date (latest params <= cutover)
    params_asof = params_df.loc[:cutover].tail(1)
    if params_asof.empty:
        # fall back to first available
        params_asof = params_df.head(1)
    params_asof = params_asof.iloc[0]

    factor_pre = factor_ret.loc[factor_ret.index < cutover]
    synth_pre = make_synthetic_returns(factor_pre, params_asof)

    port_long = splice_returns(synth_pre, port_ret_actual, cutover=cutover)

    return {
        "port_ret_long": port_long.rename("PortRet_Long"),
        "port_ret_synth_pre": synth_pre.rename("PortRet_Synth"),
        "params_rolling": params_df,
        "params_used": params_asof,
        "cutover": cutover,
    }
