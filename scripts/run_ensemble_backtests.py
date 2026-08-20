"""Run ensemble forecasts and optional backtest pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.backtest import VolTargetEngine
from src.env import Env
from src.evaluation.precision_metrics import evaluate_multiple_estimators, evaluate_vol_forecast


def _load_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def load_config(config_path: str | Path) -> dict:
    p = Path(config_path)
    if not p.exists():
        p = Env.path("strategies") / p.name
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {p}")
    return cfg


def load_data(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg.get("data", {}) or {}
    data_path = data_cfg.get("path", "data/processed/ES_Daily_Processed.parquet")
    p = Path(data_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def run_ensemble_forecast(cfg: dict) -> Path:
    est_cfg = dict(cfg.get("estimator", {}))
    class_path = est_cfg["class"]
    est_params = dict(est_cfg.get("params", {}))

    est_cls = _load_class(class_path)
    estimator = est_cls(est_params)

    df = load_data(cfg)
    forecast_df = estimator.fit_predict(df)
    run_name = str(cfg.get("name", "ensemble_run"))
    out_dir = estimator.save_artifacts(run_name)

    returns_col = est_params.get("returns_col", "returns_clean")
    if returns_col not in df.columns:
        returns_col = "returns" if "returns" in df.columns else df.columns[0]

    metrics = evaluate_vol_forecast(df[returns_col], forecast_df["ensemble_vol"])
    summary_df = pd.DataFrame([metrics], index=["ensemble"])

    comp_cols = [c for c in forecast_df.columns if c != "ensemble_vol"]
    if comp_cols:
        multi = evaluate_multiple_estimators(
            returns=df[returns_col],
            vol_forecasts={c: forecast_df[c] for c in comp_cols},
        )
        multi.to_csv(out_dir / "component_metrics.csv")

    summary_df.to_csv(out_dir / "summary_metrics.csv")

    if hasattr(estimator, "fitted_weights_"):
        pd.Series(estimator.fitted_weights_, name="weight").to_csv(out_dir / "static_weights.csv")
        with open(out_dir / "static_weights.json", "w", encoding="utf-8") as f:
            json.dump(estimator.fitted_weights_, f, indent=2, ensure_ascii=True)

    if hasattr(estimator, "regime_assignments_") and len(estimator.regime_assignments_) > 0:
        estimator.regime_assignments_.to_parquet(out_dir / "regime_assignments.parquet")
        estimator.regime_assignments_.to_csv(out_dir / "regime_assignments.csv")
        sig_cols = [c for c in estimator.regime_assignments_.columns if str(c).startswith("signal_")]
        if sig_cols:
            estimator.regime_assignments_[sig_cols].to_parquet(out_dir / "signal_table.parquet")
            estimator.regime_assignments_[sig_cols].to_csv(out_dir / "signal_table.csv")

    print(f"Ensemble artifacts saved to: {out_dir}")
    return out_dir


def run_backtest(cfg: dict):
    engine = VolTargetEngine.from_config(cfg)
    engine.run()
    engine.save()
    engine.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ensemble forecasts and backtests")
    parser.add_argument(
        "--strategy",
        default="regime_dependent_ensemble.yaml",
        help="Config path or strategy filename under configs/strategies",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Only generate ensemble forecast artifacts",
    )
    args = parser.parse_args()

    cfg = load_config(args.strategy)
    run_ensemble_forecast(cfg)

    if not args.skip_backtest:
        run_backtest(cfg)
