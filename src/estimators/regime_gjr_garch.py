from __future__ import annotations

import numpy as np
import pandas as pd

from arch import arch_model
from scipy.stats import genpareto

from src.estimators.base import Estimator

# ─── regime constants ─────────────────────────────────────────────────────────
_N_REGIMES: int = 3
_LOW: int       = 0
_MID: int       = 1
_HIGH: int      = 2


class RegimeGJRGARCH(Estimator):
    """
    Realized Regime-switching GJR-GARCH(1,1,1) volatility estimator.
    
    Regimes are determined explicitly by trailing Realized Variance (RV) 
    quantiles, bypassing the need for a latent HMM.
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = self.params

        # GJR-GARCH model spec
        self.p          = int(p.get("p",       1))
        self.q          = int(p.get("q",       1))
        self.o          = int(p.get("o",       1))
        self.dist       = str(p.get("dist",    "normal"))
        self.maxiter    = int(p.get("maxiter", 500))
        self.scale      = float(p.get("scale", 1000.0))
        self.vol_ann    = int(p.get("vol_ann", 252))

        # Window configuration
        self.lookback        = int(p.get("lookback",        252))
        self.min_obs         = int(p.get("min_obs",         126))
        self.recal_freq      = int(p.get("recal_freq",      63))

        # Realized Regime logic
        self.rv_low_q        = float(p.get("rv_low_q",      0.33))
        self.rv_high_q       = float(p.get("rv_high_q",     0.67))

        # EVT parameters
        self.use_evt         = bool(p.get("use_evt",         True))
        self.evt_tail_thresh = float(p.get("evt_tail_thresh", 0.90))
        self.evt_scale_max   = float(p.get("evt_scale_max",   1.50))

        # EWMA fallback
        self.ewma_hl = float(p.get("ewma_hl", 21))

        # ── per-regime state memory ────────────────────────────────────────────
        self._params: list[pd.Series | None] = [None, None, None]
        self._var_state: list[float]  = [np.nan, np.nan, np.nan]
        self._eps2_state: list[float] = [np.nan, np.nan, np.nan]
        self._neg_ind: list[float]    = [0.0, 0.0, 0.0]

        # ── recalibration counters ─────────────────────────────────────────────
        self._step_count: int   = 0
        self._last_recal: list[int] = [-self.recal_freq] * _N_REGIMES
        self._active_regime: int = _MID

        # ── EVT state ─────────────────────────────────────────────────────────
        self._gpd_params: tuple[float, float, float] | None = None

        # ── one-step cache ─────────────────────────────────────────────────────
        self._cache_key: tuple | None = None
        self._cache_vol: float | None = None

        # ── diagnostics ───────────────────────────────────────────────────────
        self.diag: dict = {
            "fallback_ewma":      0,    "fallback_ewma_ts":    [],
            "fallback_rv":        0,    "fallback_rv_ts":      [],
            "non_converge":       0,    "non_converge_ts":     [],
        }

    def fit(self, returns: pd.Series) -> "RegimeGJRGARCH":
        r = self._prep(returns)
        if r is None or len(r) < self.min_obs:
            return self

        for k in range(_N_REGIMES):
            r_k = self._get_regime_obs(r, k)
            if r_k is None or len(r_k) < max(20, self.min_obs // 4):
                continue
            prm = self._calibrate_gjr(r_k, k)
            if prm is not None:
                self._params[k] = prm
                init_var = float((r_k.std(ddof=1) * self.scale) ** 2)
                self._var_state[k]  = max(init_var, 1e-10)
                self._eps2_state[k] = float((r_k.iloc[-1] * self.scale) ** 2)
                self._neg_ind[k]    = float(r_k.iloc[-1] < 0)
        return self

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = self._prep(window_returns)

        if r is None or len(r) < self.min_obs:
            v = self._ewma_fallback(r)
            self.diag["fallback_ewma"] += 1
            if r is not None and len(r):
                self.diag["fallback_ewma_ts"].append(pd.Timestamp(r.index[-1]))
            return float(v) if np.isfinite(v) else np.nan

        cache_key = (r.index[-1], len(r))
        if cache_key == self._cache_key and self._cache_vol is not None:
            return float(self._cache_vol)

        self._step_count += 1
        end_ts = pd.Timestamp(r.index[-1])

        # Step 1: Detect regime purely via realized variance at t-1
        active_regime = self._detect_regime_rv(r)
        self._active_regime = active_regime

        # Step 2: Recalibrate GJR for active regime if due
        steps_since_recal = self._step_count - self._last_recal[active_regime]
        if self._params[active_regime] is None or steps_since_recal >= self.recal_freq:
            r_k = self._get_regime_obs(r, active_regime)
            if r_k is not None and len(r_k) >= max(20, self.min_obs // 4):
                prm = self._calibrate_gjr(r_k, active_regime)
                if prm is not None:
                    self._params[active_regime] = prm
                    self._last_recal[active_regime] = self._step_count
                    if not np.isfinite(self._var_state[active_regime]):
                        init_var = float((r_k.std(ddof=1) * self.scale) ** 2)
                        self._var_state[active_regime]  = max(init_var, 1e-10)
                        self._eps2_state[active_regime] = float((r_k.iloc[-1] * self.scale) ** 2)
                        self._neg_ind[active_regime] = float(r_k.iloc[-1] < 0)

        # Step 3: Advance state & forecast
        last_ret = float(r.iloc[-1])
        self._advance_regime_state(active_regime, last_ret)

        v = self._forecast(active_regime, last_ret)

        if self.use_evt and active_regime == _HIGH and np.isfinite(v) and v > 0:
            v = self._apply_evt(r, v)

        # Fallbacks
        if not np.isfinite(v) or v <= 0:
            v = self._ewma_fallback(r)
            self.diag["fallback_ewma"] += 1
            self.diag["fallback_ewma_ts"].append(end_ts)

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
        """Classify the current state using rolling RV5 quantile thresholds."""
        rv5 = r.pow(2).rolling(5, min_periods=3).mean().dropna()
        if len(rv5) < 10:
            return _MID

        rv_now     = float(rv5.iloc[-1])
        low_thresh = float(np.nanquantile(rv5.values, self.rv_low_q))
        hi_thresh  = float(np.nanquantile(rv5.values, self.rv_high_q))

        if rv_now < low_thresh:
            return _LOW
        if rv_now >= hi_thresh:
            return _HIGH
        return _MID

    def _get_regime_obs(self, r: pd.Series, regime: int) -> pd.Series | None:
        """
        Extract historical observations that fall into the specified regime,
        determined purely by rolling realized variance.
        """
        rv5 = r.pow(2).rolling(5, min_periods=3).mean().dropna()
        if len(rv5) < 10:
            # Fallback to thirds-split if not enough data
            n = len(r)
            k = n // 3
            if k < 10:
                return None
            slices = { _LOW: r.iloc[:k], _MID: r.iloc[k:2*k], _HIGH: r.iloc[2*k:] }
            return slices[regime]

        low_thresh = float(np.nanquantile(rv5.values, self.rv_low_q))
        hi_thresh  = float(np.nanquantile(rv5.values, self.rv_high_q))

        offset = len(r) - len(rv5)
        r_aligned = r.iloc[offset:]

        if regime == _LOW:
            mask = rv5 < low_thresh
        elif regime == _HIGH:
            mask = rv5 >= hi_thresh
        else:
            mask = (rv5 >= low_thresh) & (rv5 < hi_thresh)

        obs = r_aligned[mask]
        return obs if len(obs) >= 10 else None

    def _calibrate_gjr(self, r: pd.Series, regime: int) -> pd.Series | None:
        r_win    = r.iloc[-self.lookback:]
        r_scaled = r_win * self.scale
        end_ts   = pd.Timestamp(r.index[-1]) if len(r) else pd.NaT

        attempt_opts = [
            {"maxiter": self.maxiter,     "ftol": 1e-6},
            {"maxiter": self.maxiter * 2, "ftol": 1e-4},
        ]
        for opts in attempt_opts:
            try:
                am  = arch_model(
                    r_scaled,
                    vol="GARCH",
                    p=self.p, o=self.o, q=self.q,
                    dist=self.dist,
                    mean="Zero",
                    rescale=False,
                )
                res = am.fit(update_freq=0, disp="off", show_warning=False, options=opts)
                prm = res.params
                
                if prm is not None and np.all(np.isfinite(prm)):
                    alpha = float(prm.get("alpha[1]", prm.iloc[1] if len(prm)>1 else 0))
                    gamma = float(prm.get("gamma[1]", prm.iloc[2] if len(prm)>2 else 0))
                    beta  = float(prm.get("beta[1]",  prm.iloc[3] if len(prm)>3 else 0))
                    
                    if (alpha + 0.5 * gamma + beta) >= 0.9999:
                        continue
                    return prm
            except Exception:
                pass

        self.diag["non_converge"] += 1
        if pd.notna(end_ts):
            self.diag["non_converge_ts"].append(end_ts)
        return None

    def _extract_gjr_params(self, prm: pd.Series) -> tuple[float, float, float, float]:
        omega = float(prm.get("omega",    prm.iloc[0]))
        alpha = float(prm.get("alpha[1]", prm.iloc[1]))
        gamma = float(prm.get("gamma[1]", prm.iloc[2]))
        beta  = float(prm.get("beta[1]",  prm.iloc[3]))
        return omega, alpha, gamma, beta

    def _advance_regime_state(self, regime: int, ret: float) -> None:
        prm = self._params[regime]
        if prm is None:
            self._eps2_state[regime] = float((ret * self.scale) ** 2)
            self._neg_ind[regime]    = float(ret < 0)
            return

        omega, alpha, gamma, beta = self._extract_gjr_params(prm)

        eps2_prev  = self._eps2_state[regime]
        var_prev   = self._var_state[regime]
        neg_prev   = self._neg_ind[regime]

        if not np.isfinite(eps2_prev) or not np.isfinite(var_prev):
            self._eps2_state[regime] = float((ret * self.scale) ** 2)
            self._var_state[regime]  = float(omega / max(1.0 - alpha - 0.5 * gamma - beta, 1e-4))
            self._neg_ind[regime]    = float(ret < 0)
            return

        var_new = omega + (alpha + gamma * neg_prev) * eps2_prev + beta * var_prev
        self._var_state[regime]  = max(float(var_new), 1e-12)
        self._eps2_state[regime] = float((ret * self.scale) ** 2)
        self._neg_ind[regime]    = float(ret < 0)

    def _forecast(self, regime: int, last_ret: float) -> float:
        prm = self._params[regime]
        if prm is None: return np.nan

        omega, alpha, gamma, beta = self._extract_gjr_params(prm)
        var_t  = self._var_state[regime]
        
        if not np.isfinite(var_t) or var_t <= 0: return np.nan

        eps2_t = float((last_ret * self.scale) ** 2)
        neg_t  = float(last_ret < 0)

        var_next = omega + (alpha + gamma * neg_t) * eps2_t + beta * var_t
        if not np.isfinite(var_next) or var_next <= 0: return np.nan

        vol_ann = float(np.sqrt(var_next / (self.scale ** 2) * self.vol_ann))
        return vol_ann if (np.isfinite(vol_ann) and 0.0 < vol_ann < 5.0) else np.nan

    def _apply_evt(self, r: pd.Series, v_base: float) -> float:
        try:
            if self._params[_HIGH] is None: return v_base

            sq = r.pow(2).values
            tail_sample = sq[-63:] if len(sq) >= 63 else sq
            if len(tail_sample) < 10: return v_base

            threshold = float(np.nanquantile(tail_sample, self.evt_tail_thresh))
            exceedances = tail_sample[tail_sample > threshold] - threshold
            if len(exceedances) < 5: return v_base

            xi, _, sigma = genpareto.fit(exceedances, floc=0.0)
            if not (np.isfinite(xi) and np.isfinite(sigma) and sigma > 0): return v_base

            self._gpd_params = (xi, sigma, threshold)
            expected_excess = sigma / (1.0 - xi) if xi < 1.0 else sigma

            mean_var        = float(np.nanmean(tail_sample)) or 1e-14
            tail_var        = float(threshold) + float(expected_excess)
            tail_multiplier = float(np.sqrt(max(tail_var, mean_var) / mean_var))
            tail_multiplier = float(np.clip(tail_multiplier, 1.0, self.evt_scale_max))

            return float(v_base * tail_multiplier)
        except Exception:
            return v_base

    def _ewma_fallback(self, r) -> float:
        if r is None or len(r) < 2: return np.nan
        ewm = (pd.Series(r).dropna().astype(float) ** 2).ewm(halflife=self.ewma_hl, adjust=False).mean()
        v   = float(np.sqrt(ewm.iloc[-1] * self.vol_ann))
        return v if (np.isfinite(v) and v > 0) else np.nan

    def _rv_fallback(self, r) -> float:
        if r is None or len(r) < 2: return np.nan
        return float(pd.Series(r).dropna().std(ddof=1) * np.sqrt(self.vol_ann))

    def _prep(self, returns) -> pd.Series | None:
        if returns is None: return None
        r = pd.Series(returns).dropna().astype(float)
        return r if len(r) >= 2 else None