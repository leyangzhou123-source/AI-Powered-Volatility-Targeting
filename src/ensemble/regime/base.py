"""Base interface for regime recognizers."""

from __future__ import annotations

from typing import Any

import pandas as pd


class BaseRegimeRecognizer:
    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.predict(df)
