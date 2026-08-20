import numpy as np
import pandas as pd
from src.estimators.buy_and_hold import Estimator


class RealizedVol(Estimator):
    def __init__(self, params=None):
        super().__init__(params)
        self.lookback = int(self.params.get("lookback", 20))
        self.vol_ann = int(self.params.get("vol_ann", 252))

        self._rv_series = None
        self._fitted = False

        # optional rolling cache
        self._cache_end = None
        self._cache_len = None
        self._cache_vol = None

    def fit(self, returns):
        r = pd.Series(returns).dropna().astype(float)
        rolling_std = r.rolling(window=self.lookback, min_periods=1).std()
        vol = rolling_std * np.sqrt(self.vol_ann)
        self._rv_series = pd.Series(vol, index=r.index, name="rv_vol").ffill().bfill()
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

        if len(r) < self.lookback:
            v = np.nan
        else:
            v = float(r.iloc[-self.lookback:].std(ddof=1) * np.sqrt(self.vol_ann))

        self._cache_end = end
        self._cache_len = n
        self._cache_vol = v
        return v if np.isfinite(v) else np.nan

    def estimate(self, t, returns=None):
        if returns is not None:
            r = pd.Series(returns).dropna()
            if len(r) and r.index[-1] == t:
                return self.estimate_window(r)

        if self._rv_series is None:
            if returns is None:
                raise ValueError("RealizedVol not fitted and no returns provided.")
            self.fit(returns)

        if t in self._rv_series.index:
            return float(self._rv_series.loc[t])

        v = self._rv_series.sort_index().asof(t)
        return float(v) if pd.notna(v) else np.nan
