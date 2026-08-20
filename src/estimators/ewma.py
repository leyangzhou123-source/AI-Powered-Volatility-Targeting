import numpy as np
import pandas as pd
from src.estimators.buy_and_hold import Estimator


class EWMA(Estimator):
    def __init__(self, params=None):
        super().__init__(params)
        self.halflife = float(self.params.get("halflife", 20))
        self.vol_ann = int(self.params.get("vol_ann", 252))
        self.decay_factor = float(np.exp(-np.log(2.0) / self.halflife))

        self._fitted = False
        self._ewma_series = None


        self._cache_end = None
        self._cache_len = None
        self._cache_vol = None

    def fit(self, returns):
        r = pd.Series(returns).dropna().astype(float)
        sq = r * r
        ewm_var = sq.ewm(halflife=self.halflife, adjust=False).mean()
        vol = np.sqrt(ewm_var * self.vol_ann)
        self._ewma_series = pd.Series(vol, index=r.index, name="ewma_vol").ffill().bfill()
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
        v = float(self._ewma_series.iloc[-1]) if self._ewma_series is not None and len(self._ewma_series) else np.nan

        self._cache_end = end
        self._cache_len = n
        self._cache_vol = v
        return v if np.isfinite(v) else np.nan

    def estimate(self, t, returns=None):

        if returns is not None:
            r = pd.Series(returns).dropna()
            if len(r) and r.index[-1] == t:
                return self.estimate_window(r)

        if self._ewma_series is None:
            if returns is None:
                raise ValueError("EWMA not fitted and no returns provided.")
            self.fit(returns)

        if t in self._ewma_series.index:
            return float(self._ewma_series.loc[t])

        v = self._ewma_series.sort_index().asof(t)
        return float(v) if pd.notna(v) else np.nan
