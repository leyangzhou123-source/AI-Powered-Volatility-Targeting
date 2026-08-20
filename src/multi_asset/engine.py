"""Multi-asset volatility targeting engine."""

from __future__ import annotations

import importlib
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from src.env import Env
from src.multi_asset.controllers import _portfolio_vol


def _load_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_rebalance_dates(dates: pd.DatetimeIndex, rebalance_freq: str) -> set:
    rebalance_freq = str(rebalance_freq or "D").upper()
    if rebalance_freq in ("D", "DAILY", "1D"):
        return set(dates)
    schedule = pd.date_range(dates.min(), dates.max(), freq=rebalance_freq)
    return set(dates.intersection(schedule))


def _slice_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode = str(mode).lower()
    if mode not in ("all", "in_sample", "out_of_sample"):
        raise ValueError("mode must be one of: 'all', 'in_sample', 'out_of_sample'")
    if mode == "all":
        return df.copy()
    split_idx = int(len(df) * 0.5)
    return df.iloc[:split_idx].copy() if mode == "in_sample" else df.iloc[split_idx:].copy()


class MultiAssetVolTargetEngine:
    """Run a covariance estimator and portfolio controller over wide asset returns."""

    def __init__(self, name: str, estimator: Any, controller: Any, config: dict[str, Any] | None = None):
        self.name = name
        self.estimator = estimator
        self.controller = controller
        self.config = dict(config or {})
        self.result: pd.DataFrame | None = None
        self.tuning_result: dict[str, Any] = {}
        self._data_cfg = self.config.get("data", {})

    @classmethod
    def from_config(cls, cfg: dict[str, Any]):
        est_cfg = cfg.get("estimator", {})
        ctrl_cfg = cfg.get("controller", {})
        est_cls = _load_class(est_cfg["class"])
        ctrl_cls = _load_class(ctrl_cfg["class"])
        return cls(
            cfg["name"],
            est_cls(est_cfg.get("params", {})),
            ctrl_cls(ctrl_cfg.get("params", {})),
            cfg,
        )

    def _load_returns(self) -> pd.DataFrame:
        path_value = self._data_cfg.get("path", "data/processed/emm_daily_log_returns_yahoo_20220210_20260210.parquet")
        path = Env.path(path_value) if not str(path_value).endswith(".parquet") else path_value
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        assets = self._data_cfg.get("assets")
        if assets:
            df = df[list(assets)]
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").fillna(0.0)
        if df.shape[1] < 2:
            raise ValueError("Multi-asset mode requires at least two return columns.")
        return df

    def _load_risk_free_data(self) -> pd.Series:
        rf_path = Env.path("processed") / "Master_Dataset.parquet"
        if not rf_path.exists():
            return pd.Series(dtype=float)
        rf_df = pd.read_parquet(rf_path)
        rf_df.index = pd.to_datetime(rf_df.index)
        rf_df = rf_df.sort_index()
        cols = {str(c).lower(): c for c in rf_df.columns}
        col = cols.get("dgs_3mo") or cols.get("dgs3mo")
        if col is None:
            return pd.Series(dtype=float)
        annual_rate_dec = pd.to_numeric(rf_df[col], errors="coerce").ffill() / 100.0
        return np.log1p(annual_rate_dec / 252.0).rename("rf_daily_return")

    def _set_nested_params(self, estimator_params: dict[str, Any], controller_params: dict[str, Any]) -> None:
        self.estimator.params.update(estimator_params)
        if "vol_ann" in estimator_params:
            self.estimator.vol_ann = float(estimator_params["vol_ann"])
        self.controller.params.update(controller_params)
        if "max_weight" in controller_params:
            self.controller.max_weight = float(controller_params["max_weight"])
        if "long_only" in controller_params:
            self.controller.long_only = bool(controller_params["long_only"])

    def _param_grid(self, grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        if not grid:
            return [{}]
        keys = list(grid)
        return [dict(zip(keys, values)) for values in product(*[grid[k] for k in keys])]

    def tune_first_year(self, returns: pd.DataFrame) -> None:
        tune_cfg = self.config.get("tuning", {})
        if not bool(tune_cfg.get("enabled", self.config.get("auto_tune", True))):
            return
        n = min(int(tune_cfg.get("days", 252)), len(returns) - 2)
        roll_window = min(int(self.config.get("roll_window", 126)), max(40, n // 2))
        if n <= roll_window + 5:
            return

        est_base = dict(getattr(self.estimator, "params", {}))
        ctrl_base = dict(getattr(self.controller, "params", {}))
        est_grid = self._param_grid(tune_cfg.get("estimator_grid", {}))
        ctrl_grid = self._param_grid(tune_cfg.get("controller_grid", {}))
        target_vol = float(self.config.get("target_vol", 0.10))
        best_score = np.inf
        best = ({}, {})

        train = returns.iloc[:n]
        for est_params in est_grid:
            for ctrl_params in ctrl_grid:
                self.estimator.params = dict(est_base)
                self.controller.params = dict(ctrl_base)
                self._set_nested_params(est_params, ctrl_params)
                prev_w = pd.Series(0.0, index=train.columns)
                rets: list[float] = []
                vol_errors: list[float] = []
                turnover: list[float] = []
                for i in range(roll_window, len(train) - 1):
                    window = train.iloc[i - roll_window:i]
                    date = train.index[i]
                    try:
                        cov = self.estimator.estimate(date, window, train)
                        w = self.controller.compute_weights(target_vol, cov, window, date, prev_w)
                    except Exception:
                        continue
                    next_ret = float(w.reindex(train.columns).fillna(0.0).dot(train.iloc[i + 1]))
                    rets.append(next_ret)
                    vol_errors.append(abs(_portfolio_vol(w, cov) - target_vol))
                    turnover.append(float((w - prev_w).abs().sum()))
                    prev_w = w
                if len(rets) < 20:
                    continue
                r = pd.Series(rets)
                ann_ret = float(r.mean() * 252)
                ann_vol = float(r.std(ddof=1) * np.sqrt(252))
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
                score = float(np.mean(vol_errors)) + 0.02 * float(np.mean(turnover)) - 0.01 * sharpe
                if score < best_score:
                    best_score = score
                    best = (dict(est_params), dict(ctrl_params))

        self.estimator.params = est_base
        self.controller.params = ctrl_base
        self._set_nested_params(*best)
        self.tuning_result = {
            "enabled": True,
            "train_days": n,
            "score": best_score,
            "estimator_params": best[0],
            "controller_params": best[1],
        }

    def run(self, mode: str = "all") -> pd.DataFrame:
        returns = _slice_mode(self._load_returns(), mode)
        self.tune_first_year(returns)

        cfg = self.config
        target_vol = float(cfg.get("target_vol", 0.10))
        cost_bps = float(cfg.get("cost_bps", 5.0))
        initial_capital = float(cfg.get("initial_capital", 1000.0))
        rebalance_freq = cfg.get("rebalance_freq", "D")
        roll_window = int(cfg.get("roll_window", 126))
        dates = returns.index
        assets = list(returns.columns)
        rebal_dates = _get_rebalance_dates(dates, rebalance_freq)
        rf_daily = self._load_risk_free_data().reindex(dates).ffill().fillna(0.0)

        rows: list[dict[str, Any]] = []
        prev_w = pd.Series(0.0, index=assets)
        current_w = prev_w.copy()
        strategy_returns = np.zeros(len(returns), dtype=float)
        equity = initial_capital

        for i in range(roll_window, len(returns) - 1):
            date = dates[i]
            window = returns.iloc[i - roll_window:i]
            cov = self.estimator.estimate(date, window, returns)
            if date in rebal_dates:
                current_w = self.controller.compute_weights(target_vol, cov, window, date, prev_w)
                current_w = current_w.reindex(assets).fillna(0.0)
            gross = float(current_w.abs().sum())
            cash_weight = 1.0 - gross
            turnover = float((current_w - prev_w).abs().sum())
            cost = turnover * (cost_bps / 10000.0)
            next_asset_returns = returns.iloc[i + 1]
            risky_ret = float(current_w.dot(next_asset_returns))
            cash_ret = float(cash_weight * rf_daily.iloc[i + 1])
            total_ret = risky_ret + cash_ret - cost
            strategy_returns[i + 1] = total_ret
            equity *= float(np.exp(total_ret))
            realized_proxy = float(window.dot(current_w).std(ddof=1) * np.sqrt(252)) if gross > 0 else 0.0
            estimated_vol = _portfolio_vol(current_w, cov)

            row = {
                "date": dates[i + 1],
                "returns_with_rf": total_ret,
                "returns_no_rf": risky_ret - cost,
                "equity_curve_with_rf": equity,
                "equity_curve_no_rf": np.nan,
                "rf_daily_return": float(rf_daily.iloc[i + 1]),
                "risky_leg_return": risky_ret,
                "rf_leg_return": cash_ret,
                "turnover_cost": cost,
                "turnover": turnover,
                "gross_exposure": gross,
                "cash_weight": cash_weight,
                "estimated_portfolio_vol": estimated_vol,
                "realized_window_vol": realized_proxy,
                "vol_estimate": estimated_vol,
                "target_vol": target_vol,
            }
            for asset in assets:
                row[f"weight_{asset}"] = float(current_w[asset])
                row[f"return_{asset}"] = float(next_asset_returns[asset])
            rows.append(row)
            prev_w = current_w.copy()

        result = pd.DataFrame(rows).set_index("date")
        result["equity_curve_with_rf"] = initial_capital * np.exp(result["returns_with_rf"].fillna(0.0).cumsum())
        result["equity_curve_no_rf"] = initial_capital * np.exp(result["returns_no_rf"].fillna(0.0).cumsum())
        for key, value in self.tuning_result.items():
            if key.endswith("params"):
                result[f"tuning_{key}"] = str(value)
            else:
                result[f"tuning_{key}"] = value
        result.index.name = "date"
        self.result = result
        return result

    def save(self):
        if self.result is None:
            raise ValueError("No results to save. Run the backtest first.")
        path = Env.path("results") / f"{self.name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.result.to_parquet(path)
        return path
