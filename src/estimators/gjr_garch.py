"""
GJR-GARCH Volatility Estimator (with diagnostics)
"""

import numpy as np
import pandas as pd
from arch import arch_model


class GJRGARCH:
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            config = args[0]
        else:
            config = kwargs

        self.lookback   = int(config.get("lookback", 252))
        self.min_obs    = int(config.get("min_obs", 126))
        self.vol_ann    = int(config.get("vol_ann", 252))
        self.recal_freq = int(config.get("recal_freq", 63))
        self.scale      = float(config.get("scale", 1000.0))
        self.p          = int(config.get("p", 1))
        self.q          = int(config.get("q", 1))
        self.o          = int(config.get("o", 1))
        self.dist       = str(config.get("dist", "normal"))
        self.maxiter    = int(config.get("maxiter", 500))
        self.ewma_hl    = float(config.get("ewma_hl", 21))

        self._current_params = None
        self._step_count     = 0
        self._last_recal     = -self.recal_freq

        self._cache_key = None
        self._cache_vol = None

        # -----------------------------
        # Diagnostics (engine-friendly)
        # -----------------------------
        self.diag = {
            "fallback_ewma": 0,
            "fallback_rv": 0,
            "non_converge": 0,
            "warmup_insufficient": 0,

            "fallback_ewma_ts": [],
            "fallback_rv_ts": [],
            "non_converge_ts": [],
            "warmup_insufficient_ts": [],
        }

    # ------------------------------------------------------------------ #
    # Public interface                                                    #
    # ------------------------------------------------------------------ #

    def fit(self, returns: pd.Series):
        r = self._prep(returns)
        if r is None or len(r) < self.min_obs:
            return self

        r_win  = r.iloc[-self.lookback:]
        params = self._calibrate(r_win)  # _calibrate will log non_converge on failure
        if params is not None:
            self._current_params = params

        return self

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = self._prep(window_returns)

        # timestamp for this "decision point"
        end_ts = pd.Timestamp(r.index[-1]) if (r is not None and len(r)) else pd.NaT

        # Warm-up: EWMA
        if r is None or len(r) < self.min_obs:
            self.diag["warmup_insufficient"] += 1
            if pd.notna(end_ts):
                self.diag["warmup_insufficient_ts"].append(end_ts)

            v = self._ewma_fallback(r)
            self.diag["fallback_ewma"] += 1
            if pd.notna(end_ts):
                self.diag["fallback_ewma_ts"].append(end_ts)
            return float(v) if np.isfinite(v) else np.nan

        cache_key = (r.index[-1], len(r))
        if cache_key == self._cache_key and self._cache_vol is not None:
            return float(self._cache_vol)

        self._step_count += 1

        # Recalibration
        steps_since_recal = self._step_count - self._last_recal
        if self._current_params is None or steps_since_recal >= self.recal_freq:
            r_win  = r.iloc[-self.lookback:]
            params = self._calibrate(r_win)  # logs non_converge if both passes fail
            if params is not None:
                self._current_params = params
                self._last_recal = self._step_count

        # Forecast then fallback hierarchy
        v = self._forecast(r)

        if (not np.isfinite(v)) or (v <= 0):
            v2 = self._ewma_fallback(r)
            self.diag["fallback_ewma"] += 1
            self.diag["fallback_ewma_ts"].append(end_ts)
            v = v2

        if (not np.isfinite(v)) or (v <= 0):
            v2 = self._rv_fallback(r)
            self.diag["fallback_rv"] += 1
            self.diag["fallback_rv_ts"].append(end_ts)
            v = v2

        self._cache_key = cache_key
        self._cache_vol = v
        return float(v) if np.isfinite(v) else np.nan

    def estimate(self, t, returns=None):
        if returns is not None:
            r = pd.Series(returns).dropna()
            if len(r) and r.index[-1] == t:
                return self.estimate_window(r)
        return np.nan

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _prep(self, returns) -> pd.Series | None:
        if returns is None:
            return None
        r = pd.Series(returns).dropna().astype(float)
        return r if len(r) >= 2 else None

    def _calibrate(self, r: pd.Series):
        r_scaled = r * self.scale
        end_ts = pd.Timestamp(r.index[-1]) if len(r) else pd.NaT

        attempts = [
            {"maxiter": self.maxiter,     "ftol": 1e-6},
            {"maxiter": self.maxiter * 2, "ftol": 1e-4},
        ]

        any_exception = False

        for opts in attempts:
            try:
                am = arch_model(
                    r_scaled,
                    vol="GARCH",
                    p=self.p,
                    o=self.o,
                    q=self.q,
                    dist=self.dist,
                    mean="Zero",
                    rescale=False,
                )
                res = am.fit(
                    update_freq=0,
                    disp="off",
                    show_warning=False,
                    options=opts,
                )
                if res.params is not None and np.all(np.isfinite(res.params)):
                    return res.params
            except Exception:
                any_exception = True

        # If we got here, both passes failed or params were invalid
        self.diag["non_converge"] += 1
        if pd.notna(end_ts):
            self.diag["non_converge_ts"].append(end_ts)

        return None

    def _forecast(self, r: pd.Series) -> float:
        if self._current_params is None:
            return np.nan

        r_win    = r.iloc[-self.lookback:]
        r_scaled = r_win * self.scale

        try:
            am = arch_model(
                r_scaled,
                vol="GARCH",
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                mean="Zero",
                rescale=False,
            )
            res_fixed = am.fix(self._current_params)
            forecasts = res_fixed.forecast(horizon=1, reindex=False)

            var_scaled = float(forecasts.variance["h.1"].iloc[-1])
            if (not np.isfinite(var_scaled)) or (var_scaled < 0):
                return np.nan

            v_scaled = np.sqrt(var_scaled)
            v = (v_scaled / self.scale) * np.sqrt(self.vol_ann)

            return float(v) if (np.isfinite(v) and v > 0) else np.nan

        except Exception:
            return np.nan

    def _ewma_fallback(self, r) -> float:
        if r is None or len(r) < 2:
            return np.nan
        r_s = pd.Series(r).dropna().astype(float)
        ewm = (r_s ** 2).ewm(halflife=self.ewma_hl, adjust=False).mean()
        v = float(np.sqrt(ewm.iloc[-1] * self.vol_ann))
        return v if (np.isfinite(v) and v > 0) else np.nan

    def _rv_fallback(self, r) -> float:
        if r is None or len(r) < 2:
            return np.nan
        return float(pd.Series(r).dropna().std(ddof=1) * np.sqrt(self.vol_ann))