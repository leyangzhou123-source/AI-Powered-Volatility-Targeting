"""Simple change-point recognizer using rolling distribution shifts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ensemble.regime.base import BaseRegimeRecognizer
from src.ensemble.regime.regime_labels import NORMAL, TRANSITION


class ChangePointRecognizer(BaseRegimeRecognizer):
    def __init__(self, params=None):
        super().__init__(params)
        self.lookback = int(self.params.get("lookback", 21))
        self.z_threshold = float(self.params.get("z_threshold", 2.5))

    def fit(self, df: pd.DataFrame):
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df)

        r = pd.Series(df["returns"]).astype(float)
        mu = r.rolling(self.lookback, min_periods=max(5, self.lookback // 3)).mean()
        sd = r.rolling(self.lookback, min_periods=max(5, self.lookback // 3)).std(ddof=1)
        z = (r - mu) / sd.replace(0.0, np.nan)

        cp = z.abs() >= self.z_threshold
        label = pd.Series(NORMAL, index=r.index, dtype="object")
        label[cp.fillna(False)] = TRANSITION

        out = pd.DataFrame(index=pd.to_datetime(df.index))
        out["regime_label"] = label.reindex(out.index).fillna(NORMAL)
        out["change_point_flag"] = cp.reindex(out.index).fillna(False)
        out["change_score"] = z.abs().reindex(out.index)
        return out
