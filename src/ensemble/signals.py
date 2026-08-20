"""Signal engine for regime-aware ensembles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ensemble.utils import infer_returns_series, infer_value_column, warn_once


def compute_returns(df: pd.DataFrame, price_col: str | None = None) -> pd.Series:
    if price_col and price_col in df.columns:
        prices = pd.Series(df[price_col]).astype(float)
        returns = np.log(prices).diff()
        returns.name = "returns"
        return returns
    return infer_returns_series(df)


def compute_realized_vol(
    returns: pd.Series,
    lookback: int = 21,
    ann_factor: float = 252.0,
    min_periods: int | None = None,
) -> pd.Series:
    minp = min_periods if min_periods is not None else lookback
    rv = pd.Series(returns).astype(float).rolling(lookback, min_periods=minp).std(ddof=1)
    return (rv * np.sqrt(ann_factor)).rename("realized_vol")


def volatility_level_bucket(
    realized_vol: pd.Series,
    low_quantile: float = 0.3,
    high_quantile: float = 0.7,
) -> pd.Series:
    s = pd.Series(realized_vol).astype(float)
    lo = float(s.quantile(low_quantile))
    hi = float(s.quantile(high_quantile))
    out = pd.Series("mid", index=s.index, dtype="object")
    out[s <= lo] = "low"
    out[s >= hi] = "high"
    out.name = "volatility_level"
    return out


def volatility_trend(
    realized_vol: pd.Series,
    lookback: int = 5,
    threshold: float = 0.0,
) -> pd.Series:
    s = pd.Series(realized_vol).astype(float)
    delta = s - s.shift(lookback)
    out = pd.Series("flat", index=s.index, dtype="object")
    out[delta > threshold] = "rising"
    out[delta < -abs(threshold)] = "falling"
    out.name = "volatility_trend"
    return out


def load_vix_series(path: str | Path) -> pd.Series:
    p = Path(path)
    if not p.exists():
        warn_once(f"VIX file not found: {p}")
        return pd.Series(dtype=float, name="vix")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("date")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    col = infer_value_column(
        df,
        candidates=["vix", "vix_close", "close", "value", "adj_close", "price"],
    )
    return pd.Series(df[col]).astype(float).rename("vix")


def vix_level_bucket(
    vix: pd.Series,
    low_quantile: float = 0.3,
    high_quantile: float = 0.7,
) -> pd.Series:
    s = pd.Series(vix).astype(float)
    lo = float(s.quantile(low_quantile))
    hi = float(s.quantile(high_quantile))
    out = pd.Series("mid", index=s.index, dtype="object")
    out[s <= lo] = "low"
    out[s >= hi] = "high"
    out.name = "vix_level"
    return out


def iv_rv_spread(vix: pd.Series, realized_vol: pd.Series) -> pd.Series:
    x = pd.Series(vix).astype(float)
    y = pd.Series(realized_vol).astype(float)
    spread = x - y
    spread.name = "iv_rv_spread"
    return spread


def iv_rv_spread_bucket(
    spread: pd.Series,
    threshold_quantile: float = 0.7,
) -> pd.Series:
    s = pd.Series(spread).astype(float)
    th = float(s.quantile(threshold_quantile))
    out = pd.Series("normal", index=s.index, dtype="object")
    out[s >= th] = "wide"
    out[s <= -th] = "inverted"
    out.name = "iv_rv_spread_level"
    return out


def rolling_correlation_spike(
    df: pd.DataFrame,
    lookback: int = 21,
    threshold: float = 0.8,
    columns: list[str] | None = None,
) -> pd.Series:
    cols = columns or [c for c in df.columns if str(c).startswith("returns_")]
    if len(cols) < 2:
        warn_once("Correlation spike requested but multivariate returns columns are unavailable.")
        return pd.Series(False, index=df.index, name="correlation_spike")

    mat = df[cols].astype(float)

    def _mean_abs_corr(window: pd.DataFrame) -> float:
        corr = window.corr().abs().values
        if corr.shape[0] < 2:
            return np.nan
        iu = np.triu_indices(corr.shape[0], 1)
        return float(np.nanmean(corr[iu]))

    mac = mat.rolling(lookback, min_periods=lookback).apply(
        lambda _: np.nan, raw=False
    )
    vals = []
    idx = []
    for i in range(len(mat)):
        if i + 1 < lookback:
            continue
        w = mat.iloc[i + 1 - lookback : i + 1]
        vals.append(_mean_abs_corr(w))
        idx.append(mat.index[i])
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name="mean_abs_corr")
    out = s.reindex(df.index)
    return (out >= threshold).fillna(False).rename("correlation_spike")


def liquidity_stress_proxy(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold_quantile: float = 0.8,
) -> pd.Series:
    candidates = columns or [
        "liquidity_stress",
        "spread",
        "bid_ask_spread",
        "amihud",
        "volume_imbalance",
    ]
    found = [c for c in candidates if c in df.columns]
    if not found:
        warn_once("Liquidity stress requested but no liquidity columns are available.")
        return pd.Series(False, index=df.index, name="liquidity_stress")

    s = pd.Series(df[found[0]]).astype(float)
    th = float(s.quantile(threshold_quantile))
    return (s >= th).fillna(False).rename("liquidity_stress")


def drawdown_signal(returns: pd.Series, threshold: float = 0.1) -> pd.Series:
    r = pd.Series(returns).fillna(0.0).astype(float)
    equity = np.exp(r.cumsum())
    dd = equity / equity.cummax() - 1.0
    return (dd <= -abs(threshold)).rename("drawdown_stress")
