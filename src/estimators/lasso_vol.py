import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from src.estimators.buy_and_hold import Estimator


class LassoVolatility(Estimator):
    
    def __init__(self, params=None):
        super().__init__(params)

        
        self.alpha = float(self.params.get("alpha", 0.1))
        self.lags = int(self.params.get("lags", 22))
        self.fit_intercept = bool(self.params.get("fit_intercept", True))

       
        self.vol_ann = int(self.params.get("vol_ann", 252))
        self.min_obs = int(self.params.get("min_obs", 50))
        self.fallback = str(self.params.get("fallback", "rv")).lower()

      
        self._fitted = False
        self._vol_series = None
        self._cache_end = None
        self._cache_len = None
        self._cache_vol = None

        
        self.diag = {
            "fallback_rv": 0,
            "fallback_rv_ts": [],
            "non_converge": 0,         
            "non_converge_ts": [],
            "warmup_insufficient": 0,    
            "warmup_insufficient_ts": [],
        }

 
        self._last_fit_failed = False

    @staticmethod
    def _rv_fallback(r: pd.Series, vol_ann: int) -> float:
        r = pd.Series(r).dropna().astype(float)
        if len(r) < 2:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(vol_ann))

    def fit(self, returns):
        r = pd.Series(returns).dropna().astype(float)
        end_ts = pd.Timestamp(r.index[-1]) if len(r) else pd.NaT
        self._last_fit_failed = False

        # warm-up: not enough data
        if len(r) < max(self.min_obs, self.lags + 2):
            self._vol_series = pd.Series(index=r.index, dtype=float, name="lasso_vol")
            self._fitted = True

            self.diag["warmup_insufficient"] += 1
            if pd.notna(end_ts):
                self.diag["warmup_insufficient_ts"].append(end_ts)

            return self

        try:
            # target proxy variance
            daily_variance = r ** 2

            # lag features
            df = pd.DataFrame({"y": daily_variance})
            for i in range(1, self.lags + 1):
                df[f"lag_{i}"] = daily_variance.shift(i)

            df_clean = df.dropna()
            if len(df_clean) < 10:
                raise ValueError("Not enough data to fit Lasso after dropping NaNs.")

            X = df_clean.drop("y", axis=1)
            y = df_clean["y"]

            # fit Lasso
            model = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                positive=True,
                max_iter=2000,
            )
            model.fit(X, y)

            # predict variance on the available rows
            pred_var = model.predict(X)

            # strictly positive variance
            pred_var = np.maximum(pred_var, 1e-8)

            # annualize vol
            vol_ann_values = np.sqrt(pred_var) * np.sqrt(self.vol_ann)

            # align index
            pred_series = pd.Series(vol_ann_values, index=df_clean.index)
            self._vol_series = pred_series.reindex(r.index).ffill()

            # last value must be finite positive
            v_end = float(self._vol_series.iloc[-1]) if len(self._vol_series) else np.nan
            if (not np.isfinite(v_end)) or (v_end <= 0):
                self._last_fit_failed = True
                self.diag["non_converge"] += 1
                if pd.notna(end_ts):
                    self.diag["non_converge_ts"].append(end_ts)

            self._fitted = True
            return self

        except Exception:
            self._vol_series = pd.Series(index=r.index, dtype=float, name="lasso_vol")
            self._fitted = True
            self._last_fit_failed = True

            self.diag["non_converge"] += 1
            if pd.notna(end_ts):
                self.diag["non_converge_ts"].append(end_ts)

            return self

    def estimate_window(self, window_returns: pd.Series) -> float:
        """Calculate single point estimate for a rolling window (uses cache)."""
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

        # fallback
        if (not np.isfinite(v)) or (v <= 0):
            if self.fallback == "rv":
                v2 = self._rv_fallback(r, self.vol_ann)

                # count fallback usage 
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