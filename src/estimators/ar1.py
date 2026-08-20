import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.estimators.base import Estimator


class AR1(Estimator):
    def __init__(self, params=None):
        super().__init__(params)
        self.vol_ann = int(self.params.get("vol_ann", 252))

        self._vol_series = None
        self._fitted = False
        self.alpha = None
        self.beta = None


        self._cache_end = None
        self._cache_len = None
        self._cache_vol = None

    def fit(self, returns):
        r = pd.Series(returns).dropna().astype(float)
        sq = (r * r).dropna()
        if len(sq) < 3:
            self._vol_series = pd.Series(index=r.index, dtype=float, name="ar1_vol")
            self._fitted = True
            self.alpha, self.beta = np.nan, np.nan
            return self

        y = sq.iloc[1:].values.reshape(-1, 1)
        X = sq.iloc[:-1].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        self.alpha = float(model.intercept_[0])
        self.beta = float(model.coef_[0][0])

        cond_var = self.alpha + self.beta * sq.shift(1)
        cond_var = cond_var.clip(lower=1e-9)
        vol = np.sqrt(cond_var * self.vol_ann)

        self._vol_series = pd.Series(vol, index=sq.index, name="ar1_vol").ffill().bfill()
        self._fitted = True
        return self

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = pd.Series(window_returns).dropna().astype(float)
        if len(r) == 0:
            return np.nan

        end = r.index[-1]
        n = len(r)
        if self._cache_end == end and self._cache_len == n and self._cache_vol is not None:
            return float(self._cache_vol)

     
        self.fit(r)
        v = np.nan
        if self._vol_series is not None and len(self._vol_series) > 0:
            v = float(self._vol_series.iloc[-1])

        self._cache_end = end
        self._cache_len = n
        self._cache_vol = v
        return v if np.isfinite(v) else np.nan

    def estimate(self, t, returns=None):
        if returns is not None:
            r = pd.Series(returns).dropna()
            if len(r) and r.index[-1] == t:
                return self.estimate_window(r)

        if self._vol_series is None:
            if returns is None:
                raise ValueError("AR1 model has not been fitted.")
            self.fit(returns)

        if t in self._vol_series.index:
            return float(self._vol_series.loc[t])

        v = self._vol_series.sort_index().asof(t)
        return float(v) if pd.notna(v) else np.nan