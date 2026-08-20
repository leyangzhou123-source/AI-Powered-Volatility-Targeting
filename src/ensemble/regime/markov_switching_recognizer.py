"""Statsmodels Markov switching regime recognizer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ensemble.regime.base import BaseRegimeRecognizer
from src.ensemble.regime.regime_labels import HIGH, LOW, NORMAL
from src.ensemble.signals import compute_realized_vol


class MarkovSwitchingRecognizer(BaseRegimeRecognizer):
    def __init__(self, params=None):
        super().__init__(params)
        self.k_regimes = int(self.params.get("k_regimes", 2))
        self.switching_variance = bool(self.params.get("switching_variance", True))
        self._fit_res = None
        self._state_to_label: dict[int, str] = {}

    def _fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        rv = compute_realized_vol(df["returns"]).reindex(df.index)
        th = float(rv.quantile(0.65))
        label = pd.Series(LOW, index=df.index, dtype="object")
        label[rv >= th] = HIGH
        return pd.DataFrame({"regime_label": label.fillna(NORMAL)})

    def fit(self, df: pd.DataFrame):
        r = pd.Series(df["returns"]).astype(float).dropna()
        if len(r) < 80:
            self._fit_res = None
            self._fitted = True
            return self

        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

            mod = MarkovRegression(
                r,
                k_regimes=self.k_regimes,
                trend="c",
                switching_variance=self.switching_variance,
            )
            res = mod.fit(disp=False)
            self._fit_res = res

            p = res.smoothed_marginal_probabilities
            if isinstance(p, pd.DataFrame):
                vols = {}
                rv = compute_realized_vol(r)
                for k in p.columns:
                    w = p[k].reindex(rv.index).fillna(0.0)
                    vols[int(k)] = float((w * rv).sum() / max(w.sum(), 1e-12))
                order = [k for k, _ in sorted(vols.items(), key=lambda x: x[1])]
                palette = [LOW, HIGH] if self.k_regimes <= 2 else [LOW, NORMAL, HIGH]
                self._state_to_label = {s: palette[min(i, len(palette) - 1)] for i, s in enumerate(order)}
        except Exception:
            self._fit_res = None

        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df)
        if self._fit_res is None:
            return self._fallback(df)

        probs = self._fit_res.smoothed_marginal_probabilities
        if not isinstance(probs, pd.DataFrame):
            return self._fallback(df)

        probs = probs.copy()
        probs.index = pd.to_datetime(probs.index)

        out = pd.DataFrame(index=pd.to_datetime(df.index))
        for c in probs.columns:
            lbl = self._state_to_label.get(int(c), f"state_{c}")
            out[f"regime_prob_{lbl}"] = probs[c]

        best_state = probs.idxmax(axis=1).astype(int)
        out["regime_label"] = best_state.map(self._state_to_label).fillna(NORMAL)
        return out.reindex(pd.to_datetime(df.index))
