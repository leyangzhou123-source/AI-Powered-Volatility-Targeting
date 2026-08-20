"""Rule-based regime-dependent ensemble volatility estimator."""

from __future__ import annotations

import pandas as pd

from src.ensemble.base import BaseEnsembleEstimator
from src.ensemble.registry import create_recognizer
from src.ensemble.utils import normalize_weights, save_table


class RegimeDependentEnsemble(BaseEnsembleEstimator):
    """sigma_hat_t = sum_i w_i(regime_t) * sigma_hat_i,t."""

    def __init__(self, params=None):
        super().__init__(params)
        self.recognizer_cfg = dict(self.params.get("recognizer", {}))
        self.regime_weights = dict(self.params.get("regimes", {}))
        self.default_regime = str(self.params.get("default_regime", "normal"))
        self.recognizer = create_recognizer(self.recognizer_cfg)

        self.regime_assignments_: pd.DataFrame = pd.DataFrame()

    def _weights_for_regime(self, regime: str) -> dict[str, float]:
        cfg = self.regime_weights.get(regime, self.regime_weights.get(self.default_regime, {}))
        if isinstance(cfg, dict) and "weights" in cfg:
            cfg = cfg["weights"]
        return normalize_weights(dict(cfg or {}), self.estimator_names)

    def _combine_hard_label(
        self,
        component_table: pd.DataFrame,
        regime_df: pd.DataFrame,
    ) -> pd.Series:
        out = pd.Series(index=component_table.index, dtype=float, name="ensemble_vol")
        labels = regime_df.get("regime_label", pd.Series(index=component_table.index, dtype="object"))

        for t in component_table.index:
            label = str(labels.loc[t]) if t in labels.index and pd.notna(labels.loc[t]) else self.default_regime
            w = pd.Series(self._weights_for_regime(label))
            row = component_table.loc[t]
            out.loc[t] = float((row * w).sum(skipna=True))

        return out

    def _combine_soft_scores(
        self,
        component_table: pd.DataFrame,
        regime_df: pd.DataFrame,
    ) -> pd.Series:
        out = pd.Series(index=component_table.index, dtype=float, name="ensemble_vol")
        prob_cols = [c for c in regime_df.columns if str(c).startswith("regime_prob_")]
        if not prob_cols:
            return self._combine_hard_label(component_table, regime_df)

        for t in component_table.index:
            p = regime_df.loc[t, prob_cols] if t in regime_df.index else pd.Series(dtype=float)
            p = p.dropna().astype(float)
            if p.empty:
                continue

            agg_w = pd.Series(0.0, index=self.estimator_names)
            for c, prob in p.items():
                regime = str(c).replace("regime_prob_", "")
                agg_w = agg_w + float(prob) * pd.Series(self._weights_for_regime(regime))
            s = float(agg_w.sum())
            if s > 0:
                agg_w = agg_w / s

            out.loc[t] = float((component_table.loc[t] * agg_w).sum(skipna=True))

        return out

    def combine_component_forecasts(
        self,
        component_table: pd.DataFrame,
        window_returns: pd.Series | None = None,
    ) -> pd.Series:
        if window_returns is None:
            empty = pd.DataFrame(index=component_table.index, columns=["regime_label"]) 
            self.regime_assignments_ = empty
            return self._combine_hard_label(component_table, empty)

        in_df = pd.DataFrame({"returns": pd.Series(window_returns).astype(float)})
        regime_df = self.recognizer.fit_predict(in_df)
        regime_df = regime_df.reindex(component_table.index)
        self.regime_assignments_ = regime_df.copy()

        ens = self._combine_soft_scores(component_table, regime_df)

        return ens.rename("ensemble_vol")

    def save_artifacts(self, run_name: str | None = None):
        out_dir = super().save_artifacts(run_name=run_name)
        if not self.regime_assignments_.empty:
            save_table(self.regime_assignments_, out_dir / "regime_assignments.parquet")
            save_table(self.regime_assignments_, out_dir / "regime_assignments.csv")
            signal_cols = [c for c in self.regime_assignments_.columns if c.startswith("signal_")]
            if signal_cols:
                save_table(
                    self.regime_assignments_[signal_cols],
                    out_dir / "signal_table.parquet",
                )
                save_table(
                    self.regime_assignments_[signal_cols],
                    out_dir / "signal_table.csv",
                )
        return out_dir
