from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimators.base import Estimator

# ─── regime constants ─────────────────────────────────────────────────────────
_N_REGIMES: int = 3
_LOW: int       = 0
_MID: int       = 1
_HIGH: int      = 2


class RSHARRates(Estimator):
    """
    Realized Regime-Switching HAR-RV with independent per-regime Ridge regression.
    
    Regimes are determined explicitly by trailing Realized Variance (RV) 
    quantiles, bypassing the need for a latent HMM. Features are built
    continuously to preserve the chronological integrity of the HAR lags.
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = self.params

        self.w_lookback      = int(p.get("w_lookback",     5))
        self.m_lookback      = int(p.get("m_lookback",     22))
        self.ridge_lambda    = float(p.get("ridge_lambda", 1e-4))
        self.min_obs         = int(p.get("min_obs",        80))
        self.vol_ann         = int(p.get("vol_ann",        252))
        self.lookback        = int(p.get("lookback",       252))
        self.recal_freq      = int(p.get("recal_freq",     21))

        # Realized Regime Logic
        self.rv_low_q        = float(p.get("rv_low_q",      0.33))
        self.rv_high_q       = float(p.get("rv_high_q",     0.67))

        # ── per-regime HAR coefficient vectors ────────────────────────────────
        # beta[k] = np.ndarray([beta_0, beta_d, beta_w, beta_m]) or None
        self._beta: list[np.ndarray | None] = [None, None, None]

        # ── recalibration counters ─────────────────────────────────────────────
        self._step_count: int     = 0
        self._last_recal: list[int] = [-self.recal_freq] * _N_REGIMES
        self._active_regime: int  = _MID

        # ── one-step cache ─────────────────────────────────────────────────────
        self._cache_key: tuple | None = None
        self._cache_vol: float | None = None

        # ── diagnostics ───────────────────────────────────────────────────────
        self.diag: dict = {
            "fallback_rv":        0,    "fallback_rv_ts":       [],
            "non_converge":       0,    "non_converge_ts":      [],
        }

    def fit(self, returns: pd.Series) -> "RSHARRates":
        r = self._prep(returns)
        if r is None or len(r) < self.min_obs:
            return self
        
        for k in range(_N_REGIMES):
            beta = self._calibrate_har(r, k)
            if beta is not None:
                self._beta[k] = beta
        return self

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = self._prep(window_returns)

        if r is None or len(r) < self.min_obs:
            v = self._rv_fallback(r)
            self.diag["fallback_rv"] += 1
            if r is not None and len(r):
                self.diag["fallback_rv_ts"].append(pd.Timestamp(r.index[-1]))
            return float(v) if np.isfinite(v) else np.nan

        cache_key = (r.index[-1], len(r))
        if cache_key == self._cache_key and self._cache_vol is not None:
            return float(self._cache_vol)

        self._step_count += 1
        end_ts = pd.Timestamp(r.index[-1])

        # Step 1: Detect regime via trailing realized variance
        active = self._detect_regime_rv(r)
        self._active_regime = active

        # Step 2: Recalibrate HAR for active regime if due
        steps_since = self._step_count - self._last_recal[active]
        if self._beta[active] is None or steps_since >= self.recal_freq:
            beta = self._calibrate_har(r, active)
            if beta is not None:
                self._beta[active] = beta
                self._last_recal[active] = self._step_count

        # Step 3: Compute current HAR inputs for t
        rv_d, rv_w, rv_m = self._compute_har_inputs(r)

        # Step 4: Forecast t+1
        v = self._forecast(active, rv_d, rv_w, rv_m)

        # Step 5: Fallback
        if not np.isfinite(v) or v <= 0:
            v = self._rv_fallback(r)
            self.diag["fallback_rv"] += 1
            self.diag["fallback_rv_ts"].append(end_ts)

        result = float(v) if np.isfinite(v) else np.nan
        self._cache_key = cache_key
        self._cache_vol = result
        return result

    def estimate(self, t, returns=None):
        if returns is not None:
            r = pd.Series(returns).dropna()
            if len(r) and r.index[-1] == t:
                return self.estimate_window(r)
        return np.nan

    def _detect_regime_rv(self, r: pd.Series) -> int:
        all_rv5 = r.pow(2).rolling(5, min_periods=3).mean().dropna()
        if len(all_rv5) < 10:
            return _MID
        
        rv_now     = float(all_rv5.iloc[-1])
        low_thresh = float(np.nanquantile(all_rv5.values, self.rv_low_q))
        hi_thresh  = float(np.nanquantile(all_rv5.values, self.rv_high_q))
        
        if rv_now < low_thresh: return _LOW
        if rv_now >= hi_thresh: return _HIGH
        return _MID

    def _calibrate_har(self, r: pd.Series, regime: int) -> np.ndarray | None:
        """
        Builds the HAR feature matrix continuously to preserve chronological lags,
        then masks down to the specific regime to run the Ridge regression.
        """
        r_use = r.iloc[-self.lookback:]
        end_ts = pd.Timestamp(r.index[-1]) if len(r) else pd.NaT

        # 1. Build features sequentially
        rv_d = r_use.pow(2)
        rv_w = rv_d.rolling(self.w_lookback, min_periods=self.w_lookback).mean()
        rv_m = rv_d.rolling(self.m_lookback, min_periods=self.m_lookback).mean()
        
        # We need a regime signal for every historical point to filter the training set
        rv5 = rv_d.rolling(5, min_periods=3).mean()

        df = pd.DataFrame({
            "rv_d": rv_d,
            "rv_w": rv_w,
            "rv_m": rv_m,
            "rv5":  rv5,
        }).dropna()

        if len(df) < 20:
            return None

        # Target = one-step-ahead RV_d
        df["y"] = df["rv_d"].shift(-1)
        df = df.dropna()

        # 2. Extract observations matching the target regime
        low_thresh = float(np.nanquantile(df["rv5"].values, self.rv_low_q))
        hi_thresh  = float(np.nanquantile(df["rv5"].values, self.rv_high_q))

        if regime == _LOW:
            mask = df["rv5"] < low_thresh
        elif regime == _HIGH:
            mask = df["rv5"] >= hi_thresh
        else:
            mask = (df["rv5"] >= low_thresh) & (df["rv5"] < hi_thresh)

        df_regime = df[mask]

        if len(df_regime) < 20:
            return None

        # 3. Fit Ridge Regression
        X_raw = df_regime[["rv_d", "rv_w", "rv_m"]]
        y     = df_regime["y"]

        X1    = np.column_stack([np.ones(len(X_raw)), X_raw.values])
        y_arr = y.values.astype(float)

        lam   = self.ridge_lambda
        I_pen = np.eye(X1.shape[1])
        I_pen[0, 0] = 0.0  # Do not penalise the intercept

        try:
            A    = X1.T @ X1 + lam * I_pen
            b    = X1.T @ y_arr
            beta = np.linalg.solve(A, b)
            if not np.all(np.isfinite(beta)):
                return None
            return beta
        except np.linalg.LinAlgError:
            self.diag["non_converge"] += 1
            if pd.notna(end_ts):
                self.diag["non_converge_ts"].append(end_ts)
            return None

    def _compute_har_inputs(self, r: pd.Series) -> tuple[float, float, float]:
        sq   = r.pow(2)
        rv_d = float(sq.iloc[-1])
        rv_w = float(sq.iloc[-self.w_lookback :].mean()) if len(sq) >= self.w_lookback else float(sq.mean())
        rv_m = float(sq.iloc[-self.m_lookback :].mean()) if len(sq) >= self.m_lookback else float(sq.mean())
        return rv_d, rv_w, rv_m

    def _forecast(self, regime: int, rv_d: float, rv_w: float, rv_m: float) -> float:
        beta = self._beta[regime]
        if beta is None:
            return np.nan

        if not (np.isfinite(rv_d) and np.isfinite(rv_w) and np.isfinite(rv_m)):
            return np.nan

        rv_hat = float(beta[0] + beta[1] * rv_d + beta[2] * rv_w + beta[3] * rv_m)
        rv_hat = max(rv_hat, 1e-12)

        v = float(np.sqrt(rv_hat * self.vol_ann))
        return v if (np.isfinite(v) and 0.0 < v < 5.0) else np.nan

    def _rv_fallback(self, r) -> float:
        if r is None or len(r) < 2:
            return np.nan
        return float(pd.Series(r).dropna().std(ddof=1) * np.sqrt(self.vol_ann))

    def _prep(self, returns) -> pd.Series | None:
        if returns is None: return None
        r = pd.Series(returns).dropna().astype(float)
        return r if len(r) >= 2 else None