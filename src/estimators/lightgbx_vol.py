"""LightGBX-style volatility estimator (LightGBM with sklearn fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimators.base import Estimator


class LightGBXVolatility(Estimator):
    def __init__(self, params=None):
        super().__init__(params)

        self.lookback = int(self.params.get("lookback", 252))
        self.lags = int(self.params.get("lags", 22))
        self.vol_ann = int(self.params.get("vol_ann", 252))
        self.min_obs = int(self.params.get("min_obs", 120))
        self.fallback = str(self.params.get("fallback", "rv")).lower()
        self.refit_every = max(1, int(self.params.get("refit_every", 21)))

        self._cache_end = None
        self._cache_len = None
        self._cache_vol = None
        self._model = None
        self._model_backend = None
        self._model_fit_count = None

        self.diag = {
            "fallback_rv": 0,
            "fallback_rv_ts": [],
            "non_converge": 0,
            "non_converge_ts": [],
            "warmup_insufficient": 0,
            "warmup_insufficient_ts": [],
            "backend": "unknown",
        }

    @staticmethod
    def _rv_fallback(r: pd.Series, vol_ann: int) -> float:
        r = pd.Series(r).dropna().astype(float)
        if len(r) < 2:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(vol_ann))

    def _build_supervised(self, r: pd.Series):
        r = pd.Series(r).dropna().astype(float)
        rv = r.rolling(window=21, min_periods=21).std(ddof=1) * np.sqrt(self.vol_ann)

        df = pd.DataFrame({"target": rv.shift(-1)})
        sq = r**2
        absr = r.abs()
        for k in range(1, self.lags + 1):
            df[f"ret_lag_{k}"] = r.shift(k)
            df[f"sq_lag_{k}"] = sq.shift(k)
            df[f"abs_lag_{k}"] = absr.shift(k)

        df["rv_5"] = r.rolling(5).std(ddof=1) * np.sqrt(self.vol_ann)
        df["rv_21"] = r.rolling(21).std(ddof=1) * np.sqrt(self.vol_ann)
        df["rv_63"] = r.rolling(63).std(ddof=1) * np.sqrt(self.vol_ann)

        df = df.dropna()
        if len(df) < self.min_obs:
            return None, None

        X = df.drop(columns=["target"]).to_numpy(dtype=float)
        y = df["target"].to_numpy(dtype=float)
        x_last = X[-1:, :]
        return X, y, x_last

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        # Prefer LightGBM if installed.
        try:
            import lightgbm as lgb

            model = lgb.LGBMRegressor(
                n_estimators=int(self.params.get("n_estimators", 50)),
                learning_rate=float(self.params.get("learning_rate", 0.05)),
                max_depth=int(self.params.get("max_depth", -1)),
                num_leaves=int(self.params.get("num_leaves", 31)),
                random_state=int(self.params.get("random_state", 42)),
                n_jobs=int(self.params.get("n_jobs", 1)),
                verbosity=int(self.params.get("verbosity", -1)),
                force_col_wise=bool(self.params.get("force_col_wise", True)),
            )
            model.fit(X, y)
            self.diag["backend"] = "lightgbm"
            return model, "lightgbm"
        except Exception:
            pass

        # Fallback backend.
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=int(self.params.get("n_estimators", 50)),
            learning_rate=float(self.params.get("learning_rate", 0.05)),
            max_depth=int(self.params.get("gb_max_depth", 3)),
            random_state=int(self.params.get("random_state", 42)),
        )
        model.fit(X, y)
        self.diag["backend"] = "sklearn_gbr"
        return model, "sklearn_gbr"

    def _fit_predict(self, X: np.ndarray, y: np.ndarray, x_last: np.ndarray, obs_count: int) -> float:
        should_refit = (
            self._model is None
            or self._model_fit_count is None
            or obs_count - self._model_fit_count >= self.refit_every
        )
        if should_refit:
            self._model, self._model_backend = self._fit_model(X, y)
            self._model_fit_count = obs_count
        self.diag["backend"] = self._model_backend or "unknown"
        return float(self._model.predict(x_last)[0])

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = pd.Series(window_returns).dropna().astype(float)
        if len(r) == 0:
            return np.nan

        end_ts = pd.Timestamp(r.index[-1])
        n = len(r)
        if self._cache_end == end_ts and self._cache_len == n and self._cache_vol is not None:
            return float(self._cache_vol)

        if len(r) < max(self.lookback, self.min_obs):
            self.diag["warmup_insufficient"] += 1
            self.diag["warmup_insufficient_ts"].append(end_ts)
            v = self._rv_fallback(r, self.vol_ann)
            self.diag["fallback_rv"] += 1
            self.diag["fallback_rv_ts"].append(end_ts)
        else:
            r_win = r.iloc[-self.lookback:]
            X, y, x_last = self._build_supervised(r_win)
            if X is None:
                self.diag["warmup_insufficient"] += 1
                self.diag["warmup_insufficient_ts"].append(end_ts)
                v = self._rv_fallback(r_win, self.vol_ann)
                self.diag["fallback_rv"] += 1
                self.diag["fallback_rv_ts"].append(end_ts)
            else:
                try:
                    v = self._fit_predict(X, y, x_last, len(r))
                except Exception:
                    self.diag["non_converge"] += 1
                    self.diag["non_converge_ts"].append(end_ts)
                    v = self._rv_fallback(r_win, self.vol_ann)
                    self.diag["fallback_rv"] += 1
                    self.diag["fallback_rv_ts"].append(end_ts)

        if (not np.isfinite(v)) or (v <= 0):
            v = self._rv_fallback(r, self.vol_ann)

        self._cache_end = end_ts
        self._cache_len = n
        self._cache_vol = float(v) if np.isfinite(v) else np.nan
        return self._cache_vol

    def estimate(self, t, returns=None):
        if returns is None:
            return np.nan
        r = pd.Series(returns).dropna()
        if len(r) == 0:
            return np.nan
        return self.estimate_window(r)
