"""Shared helpers for ensemble estimators."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def infer_returns_series(df: pd.DataFrame, returns_col: str | None = None) -> pd.Series:
    if isinstance(df, pd.Series):
        s = pd.Series(df).astype(float)
        s.index = pd.to_datetime(s.index)
        return s.sort_index().dropna()

    candidates = [
        returns_col,
        "returns_clean",
        "asset_returns",
        "returns",
        "ret",
        "log_returns",
    ]
    for c in candidates:
        if c and c in df.columns:
            s = df[c].astype(float)
            s.index = pd.to_datetime(df.index)
            return s.sort_index().dropna()

    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(float)
        s.index = pd.to_datetime(df.index)
        return s.sort_index().dropna()

    raise KeyError(
        f"Cannot infer returns column. Available columns: {list(df.columns)}. "
        "Set returns_col in params/config."
    )


def infer_value_column(df: pd.DataFrame, candidates: list[str] | None = None) -> str:
    cands = candidates or ["close", "value", "vix", "vix_close", "adj_close", "price"]
    cols = {str(c).lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in cols:
            return str(cols[c.lower()])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return str(numeric_cols[0])

    raise KeyError(f"No numeric value column found in dataframe columns={list(df.columns)}")


def normalize_weights(weights: dict[str, float], keys: list[str], eps: float = 1e-12) -> dict[str, float]:
    raw = {k: float(weights.get(k, 0.0)) for k in keys}
    s = float(sum(max(v, 0.0) for v in raw.values()))
    if s <= eps:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: max(raw[k], 0.0) / s for k in keys}


def qlike_loss(realized_vol: pd.Series, forecast_vol: pd.Series, eps: float = 1e-12) -> pd.Series:
    rv2 = np.maximum(pd.Series(realized_vol).astype(float) ** 2, eps)
    sig2 = np.maximum(pd.Series(forecast_vol).astype(float) ** 2, eps)
    return np.log(sig2) + (rv2 / sig2)


def realized_vol_proxy(returns: pd.Series, window: int = 21, ann_factor: float = 252.0) -> pd.Series:
    r = pd.Series(returns).astype(float)
    rv = r.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(ann_factor)
    return rv.rename("realized_vol")


def align_forecasts(forecasts: dict[str, pd.Series], drop_all_na: bool = True) -> pd.DataFrame:
    table = pd.concat({k: pd.Series(v).astype(float) for k, v in forecasts.items()}, axis=1)
    table.index = pd.to_datetime(table.index)
    table = table.sort_index()
    if drop_all_na:
        table = table.dropna(how="all")
    return table


def save_table(df: pd.DataFrame, path: Path, with_index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=with_index)
    else:
        raise ValueError(f"Unsupported table extension: {path}")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def warn_once(msg: str) -> None:
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
