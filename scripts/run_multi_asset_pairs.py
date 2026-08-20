"""Run every multi-asset covariance-estimator/portfolio-controller pair."""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.backtest import VolTargetEngine
from src.env import Env
from src.multi_asset.controllers import CONTROLLER_REGISTRY
from src.multi_asset.covariance_estimators import ESTIMATOR_REGISTRY


ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "sample_covariance": {},
    "expanding_covariance": {"min_obs": 126},
    "ewma_covariance": {"halflife": 21},
    "diagonal_ewma_covariance": {"halflife": 21},
    "rolling_corr_ewma_vol": {"halflife": 21, "corr_window": 126},
    "shrunk_sample_covariance": {"shrinkage": 0.25},
    "ledoit_wolf_covariance": {},
    "downside_covariance": {"sample_blend": 0.35},
    "robust_median_covariance": {},
    "regime_switching_covariance": {"fast_halflife": 10, "slow_halflife": 63},
    "vix_scaled_covariance": {"halflife": 21},
    "pca_covariance": {"n_components": 3},
    "dynamic_blend_covariance": {},
}


CONTROLLER_DEFAULTS: dict[str, dict[str, Any]] = {
    "equal_weight": {"vol_scale": True},
    "buy_and_hold": {"vol_scale": True},
    "inverse_vol": {},
    "minimum_variance": {},
    "vol_capped_min_variance": {"asset_vol_cap": 0.30},
    "equal_risk_contribution": {"iterations": 60},
    "diversified_risk_parity": {"iterations": 60},
    "momentum_tilt": {"momentum_lookback": 63, "tilt_strength": 0.8},
    "mean_variance": {"mean_lookback": 126, "risk_aversion": 6.0},
    "regime_aware_risk_budget": {"high_vol_threshold": 0.18, "defensive_boost": 1.6},
    "drawdown_brake_portfolio": {"drawdown_trigger": 0.08, "brake_scale": 0.55},
    "hysteresis_portfolio": {"rebalance_band": 0.05},
}


def class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def build_config(est_name: str, ctrl_name: str, data_path: str, auto_tune: bool) -> dict[str, Any]:
    return {
        "name": f"{est_name}__{ctrl_name}",
        "engine_mode": "multi_asset",
        "description": "Auto-generated multi-asset volatility targeting pair",
        "data": {"path": data_path},
        "target_vol": 0.10,
        "rebalance_freq": "daily",
        "roll_window": 126,
        "weight_min": 0.0,
        "weight_max": 1.5,
        "cost_bps": 5.0,
        "initial_capital": 1000.0,
        "auto_tune": auto_tune,
        "tuning": {
            "enabled": auto_tune,
            "days": 252,
            "estimator_grid": {
                "halflife": [10, 21, 42],
                "shrinkage": [0.10, 0.25, 0.45],
            }
            if est_name in {"ewma_covariance", "shrunk_sample_covariance", "vix_scaled_covariance"}
            else {},
            "controller_grid": {
                "max_weight": [0.35, 0.45, 0.60],
                "max_gross": [1.0, 1.25, 1.5],
            },
        },
        "estimator": {
            "class": class_path(ESTIMATOR_REGISTRY[est_name]),
            "params": {"vol_ann": 252, **ESTIMATOR_DEFAULTS.get(est_name, {})},
        },
        "controller": {
            "class": class_path(CONTROLLER_REGISTRY[ctrl_name]),
            "params": {"max_weight": 0.45, "max_gross": 1.5, **CONTROLLER_DEFAULTS.get(ctrl_name, {})},
        },
    }


def _run_child(cfg: dict[str, Any], path: str, log_path: str, mode: str, queue: mp.Queue) -> None:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            engine = VolTargetEngine.from_config(cfg)
            result = engine.run(mode=mode)
            result.to_parquet(path)
        Path(log_path).write_text(buf.getvalue(), encoding="utf-8")
        queue.put({"status": "ok", "rows": int(len(result))})
    except Exception as exc:
        Path(log_path).write_text(buf.getvalue(), encoding="utf-8")
        queue.put({"status": "error", "rows": 0, "error": f"{type(exc).__name__}: {exc}"})


def run_with_timeout(cfg: dict[str, Any], path: Path, log_path: Path, mode: str, timeout_sec: int) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_run_child, args=(cfg, str(path), str(log_path), mode, queue))
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {"status": "timeout", "rows": 0, "error": f"Timed out after {timeout_sec} seconds"}
    if queue.empty():
        return {"status": "error", "rows": 0, "error": f"Process exited with code {proc.exitcode}"}
    return queue.get()


def _run_record(task: dict[str, Any], mode: str, timeout_sec: int) -> dict[str, Any]:
    cfg = task["cfg"]
    path = Path(task["path"])
    log_path = Path(task["log"])
    record = {
        "name": cfg["name"],
        "estimator": cfg["estimator"]["class"],
        "controller": cfg["controller"]["class"],
        "path": str(path),
        "log": str(log_path),
        "status": "ok",
        "rows": 0,
    }
    record.update(run_with_timeout(cfg, path, log_path, mode, timeout_sec))
    return record


def _upsert_record(records: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(record.get("name", ""))
    return [r for r in records if str(r.get("name", "")) != name] + [record]


def run_all_multi_asset_pairs(
    out_dir: str | Path,
    data_path: str,
    mode: str = "all",
    timeout_sec: int = 120,
    resume: bool = True,
    auto_tune: bool = True,
    workers: int = 1,
) -> pd.DataFrame:
    output_dir = Path(out_dir)
    if not output_dir.is_absolute():
        output_dir = Env.path("root") / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    records: list[dict[str, Any]] = []
    done: set[str] = set()
    if resume and manifest_path.exists():
        previous = pd.read_csv(manifest_path)
        records = previous.to_dict("records")
        done = {str(r["name"]) for r in records if r.get("status") == "ok" and Path(str(r.get("path", ""))).exists()}

    total = len(ESTIMATOR_REGISTRY) * len(CONTROLLER_REGISTRY)
    count = 0
    tasks: list[dict[str, Any]] = []
    for est_name in ESTIMATOR_REGISTRY:
        for ctrl_name in CONTROLLER_REGISTRY:
            count += 1
            cfg = build_config(est_name, ctrl_name, data_path, auto_tune)
            name = cfg["name"]
            path = output_dir / f"{name}.parquet"
            log_path = log_dir / f"{name}.log"
            if name in done:
                continue
            tasks.append({"ordinal": count, "total": total, "cfg": cfg, "path": str(path), "log": str(log_path)})

    workers = max(int(workers), 1)
    if workers == 1:
        for task in tasks:
            print(f"[{task['ordinal']}/{task['total']}] {task['cfg']['name']}")
            record = _run_record(task, mode, timeout_sec)
            if record["status"] != "ok":
                print(f"  failed: {record.get('error', '')}")
            records = _upsert_record(records, record)
            pd.DataFrame(records).to_csv(manifest_path, index=False)
    else:
        print(f"Running {len(tasks)} remaining pairs with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(_run_record, task, mode, timeout_sec): task
                for task in tasks
            }
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "name": task["cfg"]["name"],
                        "estimator": task["cfg"]["estimator"]["class"],
                        "controller": task["cfg"]["controller"]["class"],
                        "path": task["path"],
                        "log": task["log"],
                        "status": "error",
                        "rows": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                print(f"[{completed}/{len(tasks)}] {record['name']} -> {record['status']}")
                if record["status"] != "ok":
                    print(f"  failed: {record.get('error', '')}")
                records = _upsert_record(records, record)
                pd.DataFrame(records).to_csv(manifest_path, index=False)
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all multi-asset estimator/controller pairs.")
    parser.add_argument("--out-dir", default="results/multi_asset_pairs")
    parser.add_argument("--data-path", default="data/processed/emm_daily_log_returns_yahoo_20220210_20260210.parquet")
    parser.add_argument("--mode", default="all", choices=["all", "in_sample", "out_of_sample"])
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()
    manifest = run_all_multi_asset_pairs(
        args.out_dir,
        args.data_path,
        mode=args.mode,
        timeout_sec=args.timeout_sec,
        resume=not args.no_resume,
        auto_tune=not args.no_tune,
        workers=args.workers,
    )
    print(f"Saved {len(manifest)} records to {Path(args.out_dir) / 'manifest.csv'}")


if __name__ == "__main__":
    main()
