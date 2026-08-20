"""GARCH volatility estimator (rolling-window compatible + stabilized) + diagnostics."""

import numpy as np
import pandas as pd
from src.estimators.buy_and_hold import Estimator


class GARCH(Estimator):
    def __init__(self, params=None):
        super().__init__(params)

        self.p = int(self.params.get("p", 1))
        self.q = int(self.params.get("q", 1))
        self.vol_ann = int(self.params.get("vol_ann", 252))
        self.mean = self.params.get("mean", "Zero")
        self.dist = self.params.get("dist", "normal")
        self.scale = float(self.params.get("scale", 1000.0))
        self.maxiter = int(self.params.get("maxiter", 2000))
        self.min_obs = int(self.params.get("min_obs", 30))
        self.fallback = str(self.params.get("fallback", "rv")).lower()

        self._fitted = False
        self._vol_series = None

        self._cache_end = None
        self._cache_len = None
        self._cache_vol = None

        # -----------------------------
        # Diagnostics (engine-friendly)
        # -----------------------------
        self.diag = {
            "fallback_rv": 0,
            "fallback_rv_ts": [],
            "non_converge": 0,          # fit failed (exception) OR produced unusable result
            "non_converge_ts": [],
            "warmup_insufficient": 0,   # not enough obs to fit
            "warmup_insufficient_ts": [],
        }

        # internal: set by fit(), used to decide whether to count non_converge
        self._last_fit_failed = False

    @staticmethod
    def _rv_fallback(r: pd.Series, vol_ann: int) -> float:
        r = pd.Series(r).dropna().astype(float)
        if len(r) < 2:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(vol_ann))

    def fit(self, returns):
        """Fit GARCH(p,q) model and store conditional volatility."""
        from arch import arch_model

        r = pd.Series(returns).dropna().astype(float)
        end_ts = pd.Timestamp(r.index[-1]) if len(r) else pd.NaT
        self._last_fit_failed = False

        # Not enough data to fit: mark warmup and return empty vol series
        if len(r) < max(self.min_obs, max(self.p, self.q) + 2):
            self._vol_series = pd.Series(index=r.index, dtype=float, name="garch_vol")
            self._fitted = True

            self.diag["warmup_insufficient"] += 1
            if pd.notna(end_ts):
                self.diag["warmup_insufficient_ts"].append(end_ts)

            return self

        r_scaled = self.scale * r

        try:
            am = arch_model(
                r_scaled,
                mean=self.mean,
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )
            res = am.fit(disp="off", options={"maxiter": self.maxiter})

            # Conditional vol (daily) -> annualised
            vol_daily_scaled = res.conditional_volatility
            vol_daily = vol_daily_scaled / self.scale
            vol_ann = vol_daily * np.sqrt(self.vol_ann)

            # Store
            self._vol_series = pd.Series(vol_ann, index=r.index, name="garch_vol")
            self._fitted = True

            # If the fitted vol is garbage at the end, treat as "non_converge/unusable"
            v_end = float(self._vol_series.iloc[-1]) if len(self._vol_series) else np.nan
            if (not np.isfinite(v_end)) or (v_end <= 0):
                self._last_fit_failed = True
                self.diag["non_converge"] += 1
                if pd.notna(end_ts):
                    self.diag["non_converge_ts"].append(end_ts)

            return self

        except Exception:
            # Fit threw -> non-converge
            self._vol_series = pd.Series(index=r.index, dtype=float, name="garch_vol")
            self._fitted = True
            self._last_fit_failed = True

            self.diag["non_converge"] += 1
            if pd.notna(end_ts):
                self.diag["non_converge_ts"].append(end_ts)

            return self

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = pd.Series(window_returns).dropna().astype(float)
        if len(r) == 0:
            return np.nan

        end_ts = pd.Timestamp(r.index[-1])
        n = len(r)

        if self._cache_end == end_ts and self._cache_len == n and self._cache_vol is not None:
            return float(self._cache_vol)

        self.fit(r)

        v = np.nan
        if self._vol_series is not None and len(self._vol_series) > 0:
            v = float(self._vol_series.iloc[-1])

        # If vol invalid, apply fallback (if enabled)
        if (not np.isfinite(v)) or (v <= 0):
            if self.fallback == "rv":
                v2 = self._rv_fallback(r, self.vol_ann)

                # Count fallback usage ONLY if fallback produced a finite positive number
                # (still count it as "fallback attempt" even if NaN? if you want, remove the if)
                self.diag["fallback_rv"] += 1
                self.diag["fallback_rv_ts"].append(end_ts)

                v = v2
            else:
                v = np.nan

        self._cache_end = end_ts
        self._cache_len = n
        self._cache_vol = v
        return float(v) if np.isfinite(v) else np.nan

    def estimate(self, t, returns):
        r = pd.Series(returns).dropna().astype(float)
        if len(r) == 0:
            return np.nan

        if len(r.index) > 0 and r.index[-1] == t:
            return self.estimate_window(r)

        if (not self._fitted) or (self._vol_series is None):
            self.fit(r)

        if self._vol_series is None or len(self._vol_series) == 0:
            return np.nan

        if t in self._vol_series.index:
            v = float(self._vol_series.loc[t])
            return v if np.isfinite(v) else np.nan

        s = self._vol_series.sort_index()
        v = s.asof(t)
        return float(v) if pd.notna(v) else np.nan

