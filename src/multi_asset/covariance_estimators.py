"""Covariance estimators for multi-asset volatility targeting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _as_returns_frame(returns: pd.DataFrame) -> pd.DataFrame:
    out = returns.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return out.fillna(0.0)


def _nearest_psd(cov: pd.DataFrame, min_var: float = 1e-8) -> pd.DataFrame:
    cov = cov.copy().astype(float)
    cols = list(cov.columns)
    arr = np.asarray(cov.values, dtype=float)
    arr = np.nan_to_num((arr + arr.T) / 2.0, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return cov
    vals, vecs = np.linalg.eigh(arr)
    vals = np.maximum(vals, min_var)
    arr = (vecs * vals) @ vecs.T
    arr = (arr + arr.T) / 2.0
    return pd.DataFrame(arr, index=cols, columns=cols)


def _weighted_covariance_frame(r: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(float(weights.sum()), 1e-12)
    arr = r.values.astype(float)
    mean = weights @ arr
    centered = arr - mean
    cov = centered.T @ (centered * weights[:, None])
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


class BaseCovarianceEstimator:
    name = "base_covariance"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})
        self.vol_ann = float(self.params.get("vol_ann", TRADING_DAYS))

    def estimate(self, date: pd.Timestamp, returns_window: pd.DataFrame, full_returns: pd.DataFrame | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def _annualize(self, cov: pd.DataFrame) -> pd.DataFrame:
        return _nearest_psd(cov * self.vol_ann)


class SampleCovariance(BaseCovarianceEstimator):
    name = "sample_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        return self._annualize(r.cov())


class ExpandingCovariance(BaseCovarianceEstimator):
    name = "expanding_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        if full_returns is not None:
            r = _as_returns_frame(full_returns.loc[:date].iloc[:-1])
        else:
            r = _as_returns_frame(returns_window)
        min_obs = int(self.params.get("min_obs", 63))
        if len(r) < min_obs:
            r = _as_returns_frame(returns_window)
        return self._annualize(r.cov())


class EWMACovariance(BaseCovarianceEstimator):
    name = "ewma_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        halflife = float(self.params.get("halflife", 21))
        if len(r) < 2:
            return self._annualize(r.cov())
        decay = 0.5 ** (1.0 / max(halflife, 1e-6))
        weights = decay ** np.arange(len(r) - 1, -1, -1)
        cov = _weighted_covariance_frame(r, weights)
        return self._annualize(cov)


class DiagonalEWMACovariance(EWMACovariance):
    name = "diagonal_ewma_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        cov = super().estimate(date, returns_window, full_returns)
        return pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)


class RollingCorrelationEWMAVol(BaseCovarianceEstimator):
    name = "rolling_corr_ewma_vol"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        corr_window = int(self.params.get("corr_window", min(126, len(r))))
        corr = r.tail(corr_window).corr().fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)
        halflife = float(self.params.get("halflife", 21))
        var = r.pow(2).ewm(halflife=halflife, adjust=False).mean().iloc[-1] * self.vol_ann
        vols = np.sqrt(np.maximum(var.values, 1e-10))
        cov = corr.values * np.outer(vols, vols)
        return _nearest_psd(pd.DataFrame(cov, index=r.columns, columns=r.columns))


class ShrunkSampleCovariance(BaseCovarianceEstimator):
    name = "shrunk_sample_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        shrinkage = float(self.params.get("shrinkage", 0.25))
        sample = r.cov()
        target = pd.DataFrame(np.diag(np.diag(sample.values)), index=sample.index, columns=sample.columns)
        return self._annualize((1.0 - shrinkage) * sample + shrinkage * target)


class LedoitWolfCovariance(BaseCovarianceEstimator):
    name = "ledoit_wolf_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        try:
            from sklearn.covariance import LedoitWolf

            model = LedoitWolf().fit(r.values)
            cov = pd.DataFrame(model.covariance_, index=r.columns, columns=r.columns)
            return self._annualize(cov)
        except Exception:
            return ShrunkSampleCovariance({"shrinkage": 0.35, "vol_ann": self.vol_ann}).estimate(date, r)


class DownsideCovariance(BaseCovarianceEstimator):
    name = "downside_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        centered = r.sub(r.mean(axis=0), axis=1)
        downside = centered.mask(centered > 0.0, 0.0)
        cov = downside.cov()
        blend = float(self.params.get("sample_blend", 0.35))
        return self._annualize((1.0 - blend) * cov + blend * r.cov())


class RobustMedianCovariance(BaseCovarianceEstimator):
    name = "robust_median_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        med = r.median(axis=0)
        mad = (r - med).abs().median(axis=0).replace(0.0, np.nan)
        z = ((r - med) / (1.4826 * mad)).clip(-4.0, 4.0).fillna(0.0)
        robust = z.mul(1.4826 * mad.fillna(r.std()), axis=1) + med
        return self._annualize(robust.cov())


class RegimeSwitchingCovariance(BaseCovarianceEstimator):
    name = "regime_switching_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        fast = EWMACovariance({"halflife": self.params.get("fast_halflife", 10), "vol_ann": self.vol_ann}).estimate(date, r)
        slow = EWMACovariance({"halflife": self.params.get("slow_halflife", 63), "vol_ann": self.vol_ann}).estimate(date, r)
        realized = float(np.sqrt(np.maximum(r.mean(axis=1).tail(21).var() * self.vol_ann, 0.0)))
        threshold = float(self.params.get("high_vol_threshold", 0.18))
        alpha = float(self.params.get("fast_weight_high", 0.75 if realized >= threshold else 0.35))
        return _nearest_psd(alpha * fast + (1.0 - alpha) * slow)


class VIXScaledCovariance(EWMACovariance):
    name = "vix_scaled_covariance"

    def __init__(self, params=None):
        super().__init__(params)
        self.vix_col = str(self.params.get("vix_col", "vix_close"))
        self._vix = self._load_vix(self.params.get("vix_path", "data/processed/VIX_Daily_Processed.parquet"))

    def _load_vix(self, path_value: Any) -> pd.Series:
        path = Path(str(path_value))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return pd.Series(dtype=float)
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        cols = {str(c).lower(): c for c in df.columns}
        col = cols.get(self.vix_col.lower()) or cols.get("close") or cols.get("vix") or df.columns[0]
        return pd.to_numeric(df[col], errors="coerce").sort_index().ffill()

    def estimate(self, date, returns_window, full_returns=None):
        cov = super().estimate(date, returns_window, full_returns)
        if self._vix.empty:
            return cov
        current = float(self._vix.reindex(self._vix.index.union([pd.Timestamp(date)])).sort_index().ffill().loc[pd.Timestamp(date)])
        lookback = self._vix.loc[:pd.Timestamp(date)].tail(252)
        baseline = float(lookback.median()) if len(lookback) else current
        scale = np.clip((current / max(baseline, 1e-6)) ** 2, 0.35, 4.0)
        return _nearest_psd(cov * scale)


class PCACovariance(BaseCovarianceEstimator):
    name = "pca_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        sample = r.cov() * self.vol_ann
        vals, vecs = np.linalg.eigh(np.nan_to_num(sample.values))
        keep = max(1, min(int(self.params.get("n_components", 3)), len(vals)))
        order = np.argsort(vals)[::-1]
        mask = np.zeros_like(vals, dtype=bool)
        mask[order[:keep]] = True
        diag_floor = np.diag(np.maximum(np.diag(sample.values), 1e-8))
        arr = (vecs[:, mask] * vals[mask]) @ vecs[:, mask].T + 0.15 * diag_floor
        return _nearest_psd(pd.DataFrame(arr, index=r.columns, columns=r.columns))


class DynamicBlendCovariance(BaseCovarianceEstimator):
    name = "dynamic_blend_covariance"

    def estimate(self, date, returns_window, full_returns=None):
        r = _as_returns_frame(returns_window)
        ewma = EWMACovariance({"halflife": 21, "vol_ann": self.vol_ann}).estimate(date, r)
        shrink = LedoitWolfCovariance({"vol_ann": self.vol_ann}).estimate(date, r)
        downside = DownsideCovariance({"vol_ann": self.vol_ann}).estimate(date, r)
        stress = float(np.sqrt(np.maximum(r.mean(axis=1).tail(21).var() * self.vol_ann, 0.0)))
        stress_w = np.clip((stress - 0.08) / 0.20, 0.0, 1.0)
        return _nearest_psd((0.45 - 0.20 * stress_w) * ewma + 0.35 * shrink + (0.20 + 0.20 * stress_w) * downside)


ESTIMATOR_REGISTRY = {
    cls.name: cls
    for cls in (
        SampleCovariance,
        ExpandingCovariance,
        EWMACovariance,
        DiagonalEWMACovariance,
        RollingCorrelationEWMAVol,
        ShrunkSampleCovariance,
        LedoitWolfCovariance,
        DownsideCovariance,
        RobustMedianCovariance,
        RegimeSwitchingCovariance,
        VIXScaledCovariance,
        PCACovariance,
        DynamicBlendCovariance,
    )
}
