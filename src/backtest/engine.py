import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from src.env import Env
from src.backtest.base import Engine


def _load_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_rebalance_dates(dates: pd.DatetimeIndex, rebalance_freq: str) -> set:
    if rebalance_freq is None:
        rebalance_freq = "D"

    rebalance_freq = str(rebalance_freq).upper()

    if rebalance_freq in ("D", "DAILY", "1D"):
        return set(dates)

    schedule = pd.date_range(dates.min(), dates.max(), freq=rebalance_freq)
    return set(dates.intersection(schedule))


class VolTargetEngine(Engine):
    def __init__(self, name, estimator, controller, config=None):
        self.name = name
        self.estimator = estimator
        self.controller = controller
        self.config = config or {}
        self.result = None
        self._data_cfg = self.config.get("data", {})

    @classmethod
    def from_config(cls, cfg: dict):
        engine_mode = str(
            cfg.get("engine_mode", cfg.get("mode", cfg.get("asset_mode", "single_asset")))
        ).lower()
        if engine_mode in ("multi", "multi_asset", "multi-asset", "portfolio"):
            from src.multi_asset.engine import MultiAssetVolTargetEngine

            return MultiAssetVolTargetEngine.from_config(cfg)

        name = cfg["name"]

        est_cfg = cfg.get("estimator", {})
        est_path = est_cfg["class"]
        if "EGARCH" in est_path.upper():
            raise ValueError(f"EGARCH is disabled in this engine. Estimator requested: {est_path}")

        est_cls = _load_class(est_cfg["class"])
        estimator = est_cls(est_cfg.get("params", {}))

        ctrl_cfg = cfg.get("controller", {})
        ctrl_cls = _load_class(ctrl_cfg["class"])
        controller = ctrl_cls(ctrl_cfg.get("params", {}))

        return cls(name, estimator, controller, cfg)

    def _load_local_data(self) -> pd.DataFrame:
        data_path = self._data_cfg.get("path")
        if data_path is None:
            data_path = Env.path("processed") / "ES_Daily_Processed.parquet"
        else:
            data_path = Path(data_path)
            if not data_path.is_absolute():
                data_path = Env.path("root") / data_path

        if not pd.io.common.file_exists(str(data_path)):
            raise FileNotFoundError(
                f"Processed data file not found: {data_path}. "
                f"Set config['data']['path'] if needed."
            )

        df = pd.read_parquet(data_path)
        date_col = self._data_cfg.get("date_col")
        if date_col is None:
            for candidate in ("date", "timestamp"):
                if candidate in df.columns:
                    date_col = candidate
                    break
        if date_col is not None and date_col in df.columns:
            df = df.set_index(pd.to_datetime(df[date_col]))
        else:
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        returns_col = self._data_cfg.get("returns_col")
        if returns_col is None:
            for candidate in ("returns", "returns_clean", "log_return", "simple_return"):
                if candidate in df.columns:
                    returns_col = candidate
                    break
        if returns_col and returns_col in df.columns and "returns" not in df.columns:
            df["returns"] = df[returns_col]

        if "returns" not in df.columns:
            raise ValueError(
                f"Missing required column 'returns' in risky asset data. "
                f"Available columns: {list(df.columns)}"
            )

        df["returns"] = pd.to_numeric(df["returns"], errors="coerce")

        if "returns_clean" not in df.columns:
            df["returns_clean"] = df["returns"].copy()
        else:
            df["returns_clean"] = pd.to_numeric(df["returns_clean"], errors="coerce")

        start_date = self._data_cfg.get("start_date")
        if start_date:
            df = df.loc[pd.Timestamp(start_date) :]
        end_date = self._data_cfg.get("end_date")
        if end_date:
            df = df.loc[: pd.Timestamp(end_date)]

        return df

    def _load_risk_free_data(self) -> pd.Series:
        rf_path = Env.path("processed") / "Master_Dataset.parquet"

        if not pd.io.common.file_exists(str(rf_path)):
            raise FileNotFoundError(
                f"Risk-free dataset not found: {rf_path}. "
                f"Expected Master_Dataset.parquet under data/processed."
            )

        rf_df = pd.read_parquet(rf_path)
        rf_df.index = pd.to_datetime(rf_df.index)
        rf_df = rf_df.sort_index()

        rf_cols = {str(c).lower(): c for c in rf_df.columns}
        if "dgs_3mo" in rf_cols:
            rf_col = rf_cols["dgs_3mo"]
        elif "dgs3mo" in rf_cols:
            rf_col = rf_cols["dgs3mo"]
        else:
            raise ValueError(
                f"Could not find 3M Treasury column in Master_Dataset.parquet. "
                f"Available columns: {list(rf_df.columns)}"
            )

        annual_rate_pct = pd.to_numeric(rf_df[rf_col], errors="coerce").ffill()

        # annual percent -> annual decimal
        annual_rate_dec = annual_rate_pct / 100.0

        # approximate daily simple rate from annualized yield
        daily_simple = annual_rate_dec / 252.0

        # convert to daily log return for consistency with log-return engine
        daily_log_rf = np.log1p(daily_simple)
        daily_log_rf.name = "rf_daily_return"

        return daily_log_rf

    def _slice_mode(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        mode = str(mode).lower()
        if mode not in ("all", "in_sample", "out_of_sample"):
            raise ValueError("mode must be one of: 'all', 'in_sample', 'out_of_sample'")

        if mode == "all":
            return df.copy()

        split_idx = int(len(df) * 0.5)
        if mode == "in_sample":
            return df.iloc[:split_idx].copy()
        return df.iloc[split_idx:].copy()

    def _is_hybrid_estimator(self) -> bool:
        if getattr(self.estimator, "uses_hybrid_regime_vol", False):
            return True
        if hasattr(self.estimator, "build_vol_series"):
            return True
        if hasattr(self.estimator, "estimate_window_components"):
            return True
        if hasattr(self.estimator, "estimate_components"):
            return True
        return False

    def _compute_plain_vol_series(
        self,
        dates: pd.DatetimeIndex,
        returns_clean: pd.Series,
        window: int,
        rebal_dates: set | None = None,
    ) -> pd.Series:
        n = len(dates)
        if n <= window:
            return pd.Series(dtype=float, name="vol_estimate")

        out_vals = np.full(n - window, np.nan, dtype=float)
        out_idx = dates[window:]
        last_vol = np.nan

        has_estimate_window = hasattr(self.estimator, "estimate_window")

        for i in range(window, n):
            t = dates[i]
            j = i - window

            if rebal_dates is not None and t not in rebal_dates:
                out_vals[j] = last_vol
                continue

            w = returns_clean.iloc[i - window:i].dropna()

            if len(w) == 0:
                v = np.nan
            else:
                try:
                    if has_estimate_window:
                        v = float(self.estimator.estimate_window(w))
                    else:
                        v = float(self.estimator.estimate(t, w))
                except Exception:
                    v = np.nan

            out_vals[j] = v
            last_vol = v

        return pd.Series(out_vals, index=out_idx, name="vol_estimate")

    def run(self, mode: str = "all"):
        risky_df = self._load_local_data()
        risky_df = self._slice_mode(risky_df, mode)

        cfg = self.config
        target_vol = float(cfg.get("target_vol", 0.10))
        cost_bps = float(cfg.get("cost_bps", 5.0))
        w_min = float(cfg.get("weight_min", 0.0))
        w_max = float(cfg.get("weight_max", 1.5))
        initial_capital = float(cfg.get("initial_capital", 1000.0))
        rebalance_freq = cfg.get("rebalance_freq", "D")
        roll_window = int(cfg.get("roll_window", 252))

        dates_full = risky_df.index
        returns_raw_full = risky_df["returns"].astype(float)
        returns_clean_full = risky_df["returns_clean"].astype(float)

        rebal_dates = _get_rebalance_dates(dates_full, rebalance_freq)

        if self._is_hybrid_estimator() and hasattr(self.estimator, "build_vol_series"):
            vol_series = self.estimator.build_vol_series(
                dates=dates_full,
                returns_clean=returns_clean_full,
                window=roll_window,
                rebal_dates=rebal_dates,
                config=self.config,
            )
        else:
            vol_series = self._compute_plain_vol_series(
                dates=dates_full,
                returns_clean=returns_clean_full,
                window=roll_window,
                rebal_dates=rebal_dates,
            )

        df = risky_df.loc[vol_series.index].copy()
        df["vol_estimate"] = vol_series.astype(float).values

        rf_daily = self._load_risk_free_data()
        df["rf_daily_return"] = rf_daily.reindex(df.index).ffill().fillna(0.0)

        dates = df.index
        returns_raw = df["returns"].astype(float)
        returns_clean = df["returns_clean"].astype(float)
        rf_returns = df["rf_daily_return"].astype(float)
        vol_estimates = df["vol_estimate"].astype(float).values
        n = len(df)

        rebal_dates_trim = _get_rebalance_dates(dates, rebalance_freq)

        weights_risky = np.zeros(n, dtype=float)
        weights_rf = np.zeros(n, dtype=float)

        strategy_returns_with_rf = np.zeros(n, dtype=float)
        strategy_returns_no_rf = np.zeros(n, dtype=float)

        risky_leg_returns = np.zeros(n, dtype=float)
        rf_leg_returns = np.zeros(n, dtype=float)
        turnover_costs = np.zeros(n, dtype=float)

        prev_weight = 0.0
        equity_with_rf = initial_capital
        equity_no_rf = initial_capital

        for i, t in enumerate(dates[:-1]):
            vol_est = float(vol_estimates[i]) if i < len(vol_estimates) else np.nan

            ret_t = float(returns_clean.iloc[i])
            self.controller.update(vol_estimate=vol_est, ret=ret_t, equity=equity_with_rf)

            if t in rebal_dates_trim:
                weight = self.controller.compute_weight(target_vol, vol_est, prev_weight)
                weight = max(w_min, min(w_max, float(weight)))
            else:
                weight = prev_weight

            rf_weight = 1.0 - weight

            weights_risky[i] = weight
            weights_rf[i] = rf_weight

            turnover = abs(weight - prev_weight)
            cost = turnover * (cost_bps / 10000.0)

            next_risky_ret = float(returns_raw.iloc[i + 1])
            next_rf_ret = float(rf_returns.iloc[i + 1])

            risky_leg = weight * next_risky_ret
            rf_leg = rf_weight * next_rf_ret

            total_ret_with_rf = risky_leg + rf_leg - cost
            total_ret_no_rf = risky_leg - cost

            risky_leg_returns[i + 1] = risky_leg
            rf_leg_returns[i + 1] = rf_leg
            turnover_costs[i + 1] = cost

            strategy_returns_with_rf[i + 1] = total_ret_with_rf
            strategy_returns_no_rf[i + 1] = total_ret_no_rf

            prev_weight = weight
            equity_with_rf *= float(np.exp(total_ret_with_rf))
            equity_no_rf *= float(np.exp(total_ret_no_rf))

        weights_risky[-1] = prev_weight
        weights_rf[-1] = 1.0 - prev_weight

        equity_curve_with_rf = initial_capital * np.exp(np.cumsum(strategy_returns_with_rf))
        equity_curve_no_rf = initial_capital * np.exp(np.cumsum(strategy_returns_no_rf))

        self.result = pd.DataFrame(
            {
                "returns_with_rf": strategy_returns_with_rf,
                "returns_no_rf": strategy_returns_no_rf,
                "equity_curve_with_rf": equity_curve_with_rf,
                "equity_curve_no_rf": equity_curve_no_rf,
                "weight": weights_risky,
                "rf_weight": weights_rf,
                "vol_estimate": df["vol_estimate"].astype(float).values,
                "asset_returns": returns_raw.values,
                "asset_returns_clean": returns_clean.values,
                "rf_daily_return": rf_returns.values,
                "risky_leg_return": risky_leg_returns,
                "rf_leg_return": rf_leg_returns,
                "turnover_cost": turnover_costs,
            },
            index=dates,
        )
        self.result.index.name = "date"

        return self.result

    def save(self):
        if self.result is None:
            raise ValueError("No results to save. Run the backtest first.")

        path = Env.path("results") / f"{self.name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.result.to_parquet(path)
        return path

    def summary(self):
        if self.result is None:
            print("No results. Run the backtest first.")
            return

        for ret_col, eq_col, label in [
            ("returns_with_rf", "equity_curve_with_rf", "WITH RF"),
            ("returns_no_rf", "equity_curve_no_rf", "NO RF"),
        ]:
            r = self.result[ret_col].fillna(0.0)
            ann_ret = float(r.mean() * 252)
            ann_vol = float(r.std(ddof=1) * np.sqrt(252))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            total_return = float(self.result[eq_col].iloc[-1] / self.result[eq_col].iloc[0] - 1)

            max_dd = float(
                (self.result[eq_col] / self.result[eq_col].cummax() - 1.0).min()
            )

            print(f"\nStrategy Summary: {self.name} [{label}]")
            print(f"Total Return: {total_return:.2%}")
            print(f"Annualized Return: {ann_ret:.2%}")
            print(f"Annualized Volatility: {ann_vol:.2%}")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Max Drawdown: {max_dd:.2%}")
