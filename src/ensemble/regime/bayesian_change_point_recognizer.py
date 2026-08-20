"""Bayesian change-point recognizer with graceful fallback wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ensemble.regime.base import BaseRegimeRecognizer
from src.ensemble.regime.regime_labels import NORMAL, TRANSITION


class BayesianChangePointRecognizer(BaseRegimeRecognizer):
    def __init__(self, params=None):
        super().__init__(params)
        self.lookback = int(self.params.get("lookback", 21))
        self.posterior_threshold = float(self.params.get("posterior_threshold", 0.7))
        self._backend = "fallback"

    def fit(self, df: pd.DataFrame):
        self._fitted = True
        return self

    def _fallback_scores(self, returns: pd.Series) -> pd.Series:
        # Robust proxy for posterior-like change score: normalized shift in rolling moments.
        r = pd.Series(returns).astype(float)
        mu = r.rolling(self.lookback, min_periods=max(5, self.lookback // 3)).mean()
        sd = r.rolling(self.lookback, min_periods=max(5, self.lookback // 3)).std(ddof=1)
        z = ((r - mu) / sd.replace(0.0, np.nan)).abs()
        score = 1.0 - np.exp(-z.clip(lower=0.0))
        return score.rename("posterior_change_score")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df)

        r = pd.Series(df["returns"]).astype(float)
        idx = pd.to_datetime(df.index)

        # Optional exact backend hook can be added later when a stable dependency is selected.
        score = self._fallback_scores(r)
        label = pd.Series(NORMAL, index=idx, dtype="object")
        label.loc[score.index[score >= self.posterior_threshold]] = TRANSITION

        out = pd.DataFrame(index=idx)
        out["regime_label"] = label.reindex(idx).fillna(NORMAL)
        out["posterior_change_score"] = score.reindex(idx)
        out["diagnostic"] = f"backend={self._backend}"
        return out
