"""HMM-based volatility regime recognizer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ensemble.regime.base import BaseRegimeRecognizer
from src.ensemble.regime.regime_labels import HIGH, LOW, NORMAL
from src.ensemble.signals import compute_realized_vol


class HMMRecognizer(BaseRegimeRecognizer):
    def __init__(self, params=None):
        super().__init__(params)
        self.n_components = int(self.params.get("n_components", 3))
        self.feature_cols = list(self.params.get("feature_cols", ["returns", "realized_vol"]))
        self.random_state = int(self.params.get("random_state", 42))
        self.n_iter = int(self.params.get("n_iter", 300))
        self._model = None
        self._state_to_label: dict[int, str] = {}

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.feature_cols:
            if col in df.columns:
                out[col] = pd.Series(df[col]).astype(float)
            elif col == "realized_vol":
                out[col] = compute_realized_vol(df["returns"])
        return out.replace([np.inf, -np.inf], np.nan).dropna()

    def _fallback_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        rv = compute_realized_vol(df["returns"]).reindex(df.index)
        lo, hi = float(rv.quantile(0.3)), float(rv.quantile(0.7))
        label = pd.Series(NORMAL, index=df.index, dtype="object")
        label[rv <= lo] = LOW
        label[rv >= hi] = HIGH
        return pd.DataFrame({"regime_label": label})

    def fit(self, df: pd.DataFrame):
        x = self._build_features(df)
        if len(x) < max(30, self.n_components * 10):
            self._model = None
            self._state_to_label = {}
            self._fitted = True
            return self

        try:
            from hmmlearn.hmm import GaussianHMM

            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(x.values)

            states = model.predict(x.values)
            state_mean = pd.Series(states, index=x.index).groupby(states).apply(
                lambda idx: float(x.loc[idx.index, "realized_vol"].mean())
                if "realized_vol" in x.columns
                else float(x.loc[idx.index].mean(axis=1).mean())
            )
            order = list(state_mean.sort_values().index)
            palette = [LOW, NORMAL, HIGH]
            self._state_to_label = {
                s: palette[min(i, len(palette) - 1)] for i, s in enumerate(order)
            }
            self._model = model
        except Exception:
            self._model = None
            self._state_to_label = {}

        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df)

        x = self._build_features(df)
        if self._model is None or x.empty:
            return self._fallback_regimes(df)

        states = self._model.predict(x.values)
        probs = self._model.predict_proba(x.values)

        out = pd.DataFrame(index=df.index)
        out["regime_label"] = pd.Series(index=df.index, dtype="object")
        mapped = pd.Series(states, index=x.index).map(self._state_to_label)
        out.loc[mapped.index, "regime_label"] = mapped

        for i in range(probs.shape[1]):
            label = self._state_to_label.get(i, f"state_{i}")
            out.loc[x.index, f"regime_prob_{label}"] = probs[:, i]

        out["regime_label"] = out["regime_label"].fillna(NORMAL)
        return out
