"""Equal-weight fixed ensemble volatility estimator."""

from __future__ import annotations

import pandas as pd

from src.ensemble.base import BaseEnsembleEstimator


class FixedEnsemble(BaseEnsembleEstimator):
    """sigma_hat_t = mean_i sigma_hat_i,t."""

    def combine_component_forecasts(
        self,
        component_table: pd.DataFrame,
        window_returns=None,
    ) -> pd.Series:
        table = component_table.copy()
        ensemble = table.mean(axis=1, skipna=True)
        return ensemble.rename("ensemble_vol")
