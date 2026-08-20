"""Markov-switching GARCH recognizer with explicit approximation fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ensemble.regime.base import BaseRegimeRecognizer
from src.ensemble.regime.regime_labels import HIGH, LOW, NORMAL


class MarkovSwitchingGARCHRecognizer(BaseRegimeRecognizer):
    """
    Placeholder adapter for MS-GARCH.

    A true MS-GARCH implementation is not included here because common Python stacks
    do not provide a stable production-ready API for it. This recognizer falls back to
    an explicit approximation: fit GARCH-style rolling vol features and cluster states.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.lookback = int(self.params.get("lookback", 63))
        self.k_regimes = int(self.params.get("k_regimes", 2))
        self._diag_msg = "approximation_used: rolling-vol + clustering"

    def fit(self, df: pd.DataFrame):
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df)

        r = pd.Series(df["returns"]).astype(float)
        rv = r.rolling(self.lookback, min_periods=max(10, self.lookback // 3)).std(ddof=1) * np.sqrt(252.0)
        feat = pd.DataFrame({"rv": rv, "abs_r": r.abs()}).dropna()

        out = pd.DataFrame(index=pd.to_datetime(df.index))
        out["diagnostic"] = self._diag_msg

        if len(feat) < 40:
            label = pd.Series(NORMAL, index=out.index, dtype="object")
            out["regime_label"] = label
            return out

        try:
            from sklearn.cluster import KMeans

            k = max(2, min(self.k_regimes, 4))
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            states = km.fit_predict(feat.values)
            means = pd.Series(states, index=feat.index).groupby(states).apply(
                lambda s: float(feat.loc[s.index, "rv"].mean())
            )
            ordered = list(means.sort_values().index)
            palette = [LOW, HIGH] if k <= 2 else [LOW, NORMAL, HIGH, HIGH]
            mapping = {st: palette[min(i, len(palette) - 1)] for i, st in enumerate(ordered)}

            label = pd.Series(index=out.index, dtype="object")
            label.loc[feat.index] = pd.Series(states, index=feat.index).map(mapping)
            out["regime_label"] = label.fillna(NORMAL)
        except Exception:
            out["regime_label"] = NORMAL

        return out
