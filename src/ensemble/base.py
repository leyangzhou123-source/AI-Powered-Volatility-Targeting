"""Base class for volatility ensemble estimators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.estimators.base import Estimator
from src.ensemble.registry import create_estimator
from src.ensemble.utils import align_forecasts, infer_returns_series, save_table
from src.env import Env


class BaseEnsembleEstimator(Estimator):
    """Base class that makes ensembles pluggable as standard estimators."""

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.params = dict(params or {})

        self.estimator_names = [str(x).lower() for x in self.params.get("estimators", [])]
        if not self.estimator_names:
            raise ValueError("Ensemble requires a non-empty `estimators` list.")

        self.estimator_params = dict(self.params.get("estimator_params", {}))
        self.returns_col = self.params.get("returns_col", "returns_clean")
        self.min_obs = int(self.params.get("min_obs", 30))

        self.results_dir = Path(self.params.get("results_dir", Env.path("results") / "ensemble"))
        self.name = str(self.params.get("name", self.__class__.__name__.lower()))

        self.component_forecasts_: pd.DataFrame = pd.DataFrame()
        self.ensemble_forecast_: pd.Series = pd.Series(dtype=float, name="ensemble_vol")

        self._estimators = {
            n: create_estimator(n, self.estimator_params.get(n, {})) for n in self.estimator_names
        }

    def _extract_returns(self, data: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(data, pd.Series):
            r = pd.Series(data).astype(float)
            r.index = pd.to_datetime(r.index)
            return r.sort_index().dropna()
        return infer_returns_series(data, self.returns_col)

    def _estimate_component_window(self, est, t: pd.Timestamp, window_returns: pd.Series) -> float:
        if hasattr(est, "estimate_window"):
            try:
                return float(est.estimate_window(window_returns))
            except Exception:
                return np.nan
        try:
            return float(est.estimate(t, window_returns))
        except Exception:
            return np.nan

    def _component_forecasts_from_returns(self, returns: pd.Series) -> pd.DataFrame:
        r = pd.Series(returns).dropna().astype(float)
        if len(r) < self.min_obs:
            return pd.DataFrame(index=r.index, columns=self.estimator_names, dtype=float)

        table = pd.DataFrame(index=r.index, columns=self.estimator_names, dtype=float)
        for i, t in enumerate(r.index):
            if i + 1 < self.min_obs:
                continue
            window = r.iloc[: i + 1]
            for name, est in self._estimators.items():
                table.loc[t, name] = self._estimate_component_window(est, t, window)

        return table

    def get_component_forecasts(self) -> pd.DataFrame:
        return self.component_forecasts_.copy()

    def get_ensemble_forecast(self) -> pd.Series:
        return self.ensemble_forecast_.copy()

    def fit(self, returns: pd.Series):
        r = self._extract_returns(returns)
        self.component_forecasts_ = self._component_forecasts_from_returns(r)
        self._fitted = True
        return self

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = self._extract_returns(window_returns)
        if len(r) < self.min_obs:
            return np.nan
        c = {}
        t = r.index[-1]
        for n, est in self._estimators.items():
            c[n] = self._estimate_component_window(est, t, r)
        cdf = align_forecasts({k: pd.Series([v], index=[t]) for k, v in c.items()})
        out = self.combine_component_forecasts(cdf, window_returns=r)
        if isinstance(out, pd.Series):
            return float(out.iloc[-1])
        return float(out.iloc[-1, 0])

    def estimate(self, t, returns=None):
        if returns is None:
            return np.nan
        r = self._extract_returns(returns)
        if len(r) == 0:
            return np.nan
        return self.estimate_window(r)

    def combine_component_forecasts(
        self,
        component_table: pd.DataFrame,
        window_returns: pd.Series | None = None,
    ) -> pd.Series:
        raise NotImplementedError

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        r = self._extract_returns(df)
        comp = self._component_forecasts_from_returns(r)
        ens = self.combine_component_forecasts(comp, window_returns=r)

        self.component_forecasts_ = comp
        self.ensemble_forecast_ = pd.Series(ens, name="ensemble_vol")
        self._fitted = True

        return pd.concat([comp, self.ensemble_forecast_], axis=1)

    def save_artifacts(self, run_name: str | None = None) -> Path:
        out_dir = self.results_dir / (run_name or self.name)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.component_forecasts_.empty:
            save_table(self.component_forecasts_, out_dir / "component_forecasts.parquet")
            save_table(self.component_forecasts_, out_dir / "component_forecasts.csv")

        if len(self.ensemble_forecast_) > 0:
            df = pd.DataFrame({"ensemble_vol": self.ensemble_forecast_})
            save_table(df, out_dir / "ensemble_forecast.parquet")
            save_table(df, out_dir / "ensemble_forecast.csv")

        return out_dir
