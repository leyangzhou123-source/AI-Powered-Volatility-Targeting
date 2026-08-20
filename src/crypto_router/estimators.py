"""Crypto-native volatility estimators.

These estimators are intentionally dependency-light and compatible with the
single-asset ``VolTargetEngine``.  They use only information available before
the trading date by shifting daily/intraday inputs by one row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.env import Env
from src.estimators.base import Estimator


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    out = Path(path)
    if out.is_absolute():
        return out
    return Env.path("root") / out


def _as_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out = out.set_index(pd.to_datetime(out["date"]))
    else:
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _clean_positive(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).where(lambda x: x > 0)


class IntradayRealizedVolEstimator(Estimator):
    """Use precomputed intraday realized volatility as the next-day forecast."""

    uses_hybrid_regime_vol = True

    def __init__(self, params=None):
        super().__init__(params)
        self.path = self.params.get("path")
        self.vol_col = str(self.params.get("vol_col", "realized_volatility_annualized"))
        self.lookback = int(self.params.get("lookback", 3))
        self.min_coverage = float(self.params.get("min_coverage", 0.0))
        self.fallback_lookback = int(self.params.get("fallback_lookback", 20))
        self.vol_ann = float(self.params.get("vol_ann", self.params.get("annualization_factor", 365.0)))
        self._series: pd.Series | None = None

    def _load_series(self, config: dict | None = None) -> pd.Series:
        if self._series is not None:
            return self._series
        path = _resolve_path(self.path or ((config or {}).get("intraday_realized_vol", {}) or {}).get("path"))
        if path is None:
            raise ValueError("IntradayRealizedVolEstimator requires params.path or config intraday_realized_vol.path")
        df = _as_datetime_index(pd.read_parquet(path))
        vol_col = self.vol_col if self.vol_col in df.columns else "realized_vol"
        if vol_col not in df.columns:
            raise ValueError(f"Intraday RV file has no volatility column {self.vol_col!r}; columns={list(df.columns)}")
        vol = _clean_positive(df[vol_col])
        coverage_col = "coverage_ratio" if "coverage_ratio" in df.columns else "coverage"
        if coverage_col in df.columns and self.min_coverage > 0:
            vol = vol.where(pd.to_numeric(df[coverage_col], errors="coerce") >= self.min_coverage)
        self._series = vol.rolling(self.lookback, min_periods=1).mean().shift(1).ffill()
        self._series.name = "intraday_realized_vol_forecast"
        return self._series

    def build_vol_series(
        self,
        dates: pd.DatetimeIndex,
        returns_clean: pd.Series,
        window: int,
        rebal_dates: set | None = None,
        config: dict | None = None,
    ) -> pd.Series:
        vol = self._load_series(config).reindex(dates).ffill()
        fallback = returns_clean.rolling(self.fallback_lookback, min_periods=5).std(ddof=1) * np.sqrt(self.vol_ann)
        out = vol.combine_first(fallback.shift(1))
        return out.iloc[window:].rename("vol_estimate")

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = pd.Series(window_returns).dropna().astype(float)
        if len(r) < max(5, self.fallback_lookback):
            return np.nan
        return float(r.tail(self.fallback_lookback).std(ddof=1) * np.sqrt(self.vol_ann))

    def estimate(self, t, returns=None):
        if self._series is not None:
            v = self._series.sort_index().asof(pd.Timestamp(t))
            if pd.notna(v):
                return float(v)
        return self.estimate_window(pd.Series(returns))


class RangeBasedVolEstimator(Estimator):
    """Parkinson/Garman-Klass style OHLC volatility forecast."""

    uses_hybrid_regime_vol = True

    def __init__(self, params=None):
        super().__init__(params)
        self.lookback = int(self.params.get("lookback", 20))
        self.method = str(self.params.get("method", "garman_klass")).lower()
        self.vol_ann = float(self.params.get("vol_ann", self.params.get("annualization_factor", 365.0)))

    def _range_variance(self, df: pd.DataFrame) -> pd.Series:
        high = _clean_positive(df["high"])
        low = _clean_positive(df["low"])
        open_ = _clean_positive(df["open"])
        close = _clean_positive(df["close"])
        log_hl = np.log(high / low)
        if self.method in ("parkinson", "pk"):
            var = (log_hl**2) / (4.0 * np.log(2.0))
        else:
            log_co = np.log(close / open_)
            var = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
        return pd.Series(var, index=df.index).clip(lower=1e-18)

    def build_vol_series(
        self,
        dates: pd.DatetimeIndex,
        returns_clean: pd.Series,
        window: int,
        rebal_dates: set | None = None,
        config: dict | None = None,
    ) -> pd.Series:
        data_path = _resolve_path(((config or {}).get("data", {}) or {}).get("path"))
        if data_path is None:
            fallback = returns_clean.rolling(self.lookback, min_periods=5).std(ddof=1) * np.sqrt(self.vol_ann)
            return fallback.shift(1).iloc[window:].rename("vol_estimate")
        df = _as_datetime_index(pd.read_parquet(data_path))
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            fallback = returns_clean.rolling(self.lookback, min_periods=5).std(ddof=1) * np.sqrt(self.vol_ann)
            return fallback.shift(1).iloc[window:].rename("vol_estimate")
        var = self._range_variance(df)
        vol = np.sqrt(var.rolling(self.lookback, min_periods=5).mean() * self.vol_ann).shift(1)
        fallback = returns_clean.rolling(self.lookback, min_periods=5).std(ddof=1) * np.sqrt(self.vol_ann)
        out = vol.reindex(dates).ffill().combine_first(fallback.shift(1))
        return out.iloc[window:].rename("vol_estimate")

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = pd.Series(window_returns).dropna().astype(float)
        if len(r) < max(5, self.lookback):
            return np.nan
        return float(r.tail(self.lookback).std(ddof=1) * np.sqrt(self.vol_ann))

    def estimate(self, t, returns=None):
        return self.estimate_window(pd.Series(returns))


class CryptoCompositeVolEstimator(Estimator):
    """Robust blend of intraday RV, OHLC range vol, and EWMA return vol."""

    uses_hybrid_regime_vol = True

    def __init__(self, params=None):
        super().__init__(params)
        self.intraday = IntradayRealizedVolEstimator(params)
        self.range = RangeBasedVolEstimator(params)
        self.ewma_halflife = float(self.params.get("ewma_halflife", 10))
        self.vol_ann = float(self.params.get("vol_ann", self.params.get("annualization_factor", 365.0)))
        self.weights = self._parse_weights(self.params.get("weights", [0.50, 0.25, 0.25]))
        self.floor_vol = float(self.params.get("floor_vol", 0.001))
        self.cap_vol = float(self.params.get("cap_vol", 5.0))

    @staticmethod
    def _parse_weights(values: Iterable[float]) -> np.ndarray:
        w = np.asarray(list(values), dtype=float)
        if len(w) != 3 or not np.isfinite(w).all() or float(w.sum()) <= 0:
            return np.asarray([0.50, 0.25, 0.25], dtype=float)
        return w / float(w.sum())

    def build_vol_series(
        self,
        dates: pd.DatetimeIndex,
        returns_clean: pd.Series,
        window: int,
        rebal_dates: set | None = None,
        config: dict | None = None,
    ) -> pd.Series:
        intraday = self.intraday.build_vol_series(dates, returns_clean, 0, rebal_dates, config).reindex(dates)
        range_vol = self.range.build_vol_series(dates, returns_clean, 0, rebal_dates, config).reindex(dates)
        ewma = np.sqrt(
            returns_clean.pow(2).ewm(halflife=self.ewma_halflife, adjust=False).mean() * self.vol_ann
        ).shift(1)
        components = pd.concat([intraday, range_vol, ewma], axis=1)
        raw = components.mul(self.weights, axis=1).sum(axis=1, min_count=1)
        fallback = components.median(axis=1, skipna=True)
        out = raw.combine_first(fallback).clip(lower=self.floor_vol, upper=self.cap_vol)
        return out.iloc[window:].rename("vol_estimate")

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = pd.Series(window_returns).dropna().astype(float)
        if len(r) < 5:
            return np.nan
        rv = r.tail(min(20, len(r))).std(ddof=1) * np.sqrt(self.vol_ann)
        ewma = np.sqrt(r.pow(2).ewm(halflife=self.ewma_halflife, adjust=False).mean().iloc[-1] * self.vol_ann)
        v = float(np.nanmedian([rv, ewma]))
        return float(np.clip(v, self.floor_vol, self.cap_vol)) if np.isfinite(v) else np.nan

    def estimate(self, t, returns=None):
        return self.estimate_window(pd.Series(returns))
