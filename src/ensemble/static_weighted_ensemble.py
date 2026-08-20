"""Static-weight ensemble volatility estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ensemble.base import BaseEnsembleEstimator
from src.ensemble.utils import (
    normalize_weights,
    qlike_loss,
    realized_vol_proxy,
)


class StaticWeightedEnsemble(BaseEnsembleEstimator):
    """sigma_hat_t = sum_i w_i * sigma_hat_i,t with static weights."""

    def __init__(self, params=None):
        super().__init__(params)
        self.weight_method = str(self.params.get("weight_method", "manual")).lower()
        self.manual_weights = dict(self.params.get("manual_weights", {}))
        self.ann_factor = float(self.params.get("ann_factor", self.params.get("vol_ann", 252.0)))
        self.rv_window = int(self.params.get("rv_window", 21))
        self.train_start = self.params.get("train_start")
        self.train_end = self.params.get("train_end")

        self.fitted_weights_: dict[str, float] = {
            n: 1.0 / len(self.estimator_names) for n in self.estimator_names
        }

    def _select_train_slice(
        self,
        comp: pd.DataFrame,
        returns: pd.Series | None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        train_comp = comp.copy()
        train_ret = returns.copy() if returns is not None else None

        if self.train_start:
            start = pd.Timestamp(self.train_start)
            train_comp = train_comp.loc[train_comp.index >= start]
            if train_ret is not None:
                train_ret = train_ret.loc[train_ret.index >= start]
        if self.train_end:
            end = pd.Timestamp(self.train_end)
            train_comp = train_comp.loc[train_comp.index <= end]
            if train_ret is not None:
                train_ret = train_ret.loc[train_ret.index <= end]

        return train_comp, train_ret

    def _fit_weights_from_history(
        self,
        component_table: pd.DataFrame,
        window_returns: pd.Series | None,
    ) -> dict[str, float]:
        if self.weight_method == "manual":
            return normalize_weights(self.manual_weights, self.estimator_names)

        if window_returns is None:
            return {n: 1.0 / len(self.estimator_names) for n in self.estimator_names}

        comp, ret = self._select_train_slice(component_table, pd.Series(window_returns).astype(float))
        target = realized_vol_proxy(ret, window=self.rv_window, ann_factor=self.ann_factor)

        raw_scores: dict[str, float] = {}
        for name in self.estimator_names:
            joined = pd.concat([comp[name], target], axis=1).dropna()
            if len(joined) < 10:
                raw_scores[name] = 0.0
                continue

            pred = joined.iloc[:, 0]
            rv = joined.iloc[:, 1]

            if self.weight_method == "inverse_mse":
                err = float(np.mean((pred - rv) ** 2))
            elif self.weight_method == "inverse_qlike":
                err = float(np.mean(qlike_loss(rv, pred)))
            else:
                raise ValueError(
                    f"Unsupported weight_method={self.weight_method}. "
                    "Use manual|inverse_mse|inverse_qlike."
                )

            raw_scores[name] = 0.0 if (not np.isfinite(err) or err <= 0) else 1.0 / err

        return normalize_weights(raw_scores, self.estimator_names)

    def combine_component_forecasts(
        self,
        component_table: pd.DataFrame,
        window_returns: pd.Series | None = None,
    ) -> pd.Series:
        weights = self._fit_weights_from_history(component_table, window_returns)
        self.fitted_weights_ = dict(weights)

        w = pd.Series(weights)
        ens = component_table.mul(w, axis=1).sum(axis=1, min_count=1)

        return ens.rename("ensemble_vol")

    def save_artifacts(self, run_name: str | None = None):
        from src.ensemble.utils import save_json

        out_dir = super().save_artifacts(run_name=run_name)
        save_json(self.fitted_weights_, out_dir / "static_weights.json")
        pd.Series(self.fitted_weights_, name="weight").to_csv(out_dir / "static_weights.csv")
        return out_dir
