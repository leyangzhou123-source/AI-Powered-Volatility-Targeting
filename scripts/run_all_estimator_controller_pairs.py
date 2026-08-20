"""Run every estimator/controller pair discovered from the source folders."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.backtest import VolTargetEngine
from src.env import Env


ESTIMATOR_PARAM_DEFAULTS: dict[str, dict[str, Any]] = {
    "AR1": {"vol_ann": 252},
    "AR2": {"vol_ann": 252},
    "BuyAndHold": {"target_vol": 0.10},
    "DynamicPrecisionEnsemble": {
        "estimators": [
            {"class": "src.estimators.ewma.EWMA", "params": {"halflife": 8, "vol_ann": 252}},
            {"class": "src.estimators.realized_vol.RealizedVol", "params": {"lookback": 20, "vol_ann": 252}},
            {"class": "src.estimators.har_rv.HARRV", "params": {"vol_ann": 252}},
        ],
        "loss_window": 21,
    },
    "EWMA": {"halflife": 8, "vol_ann": 252},
    "GARCH": {"p": 1, "q": 1, "vol_ann": 252, "maxiter": 200},
    "GJRGARCH": {"p": 1, "o": 1, "q": 1, "vol_ann": 252, "maxiter": 200},
    "HARRV": {"vol_ann": 252},
    "HARRVRates": {"vol_ann": 252, "min_obs": 80},
    "HybridEWMARegime": {"vol_ann": 252},
    "LassoVolatility": {"vol_ann": 252},
    "LightGBXVolatility": {
        "lookback": 252,
        "lags": 10,
        "vol_ann": 252,
        "min_obs": 80,
        "n_estimators": 50,
    },
    "NaiveVolEstimator": {"vol_ann": 252},
    "RandomForestVolEstimator": {
        "n_estimators": 50,
        "max_depth": 8,
        "train_window": 252,
        "auto_tune": False,
        "use_hmm": False,
    },
    "RealizedVol": {"lookback": 20, "vol_ann": 252},
    "RegimeGJRGARCH": {"vol_ann": 252, "maxiter": 200, "min_obs": 126},
    "RNNVolatility": {
        "lookback": 252,
        "seq_len": 10,
        "vol_ann": 252,
        "min_obs": 80,
        "backend": "sklearn_mlp",
        "max_iter": 40,
        "epochs": 3,
    },
    "RSHARRates": {"vol_ann": 252, "min_obs": 80},
    "XGB_VIX": {
        "vol_ann": 252,
        "min_obs": 100,
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.03,
    },
}


CONTROLLER_PARAM_DEFAULTS: dict[str, dict[str, Any]] = {
    "ConstantWeight": {"weight": 1.0},
    "CVaRESTargeting": {},
    "DrawdownBrake": {},
    "DrawdownModulatedController": {},
    "HysteresisController": {},
    "NaiveScaling": {},
    "PriorityStackController": {},
    "RegimeSwitchController": {},
    "TrendFilter": {},
    "VarianceScaling": {},
    "VolTargetClip": {},
}


def _module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def _is_estimator_class(cls: type) -> bool:
    return callable(getattr(cls, "estimate_window", None)) or callable(getattr(cls, "estimate", None))


def _is_controller_class(cls: type) -> bool:
    return callable(getattr(cls, "compute_weight", None))


def discover_classes(folder: Path, predicate) -> list[dict[str, str]]:
    root = Env.path("root")
    discovered: list[dict[str, str]] = []
    for path in sorted(folder.glob("*.py")):
        if path.name in {"__init__.py", "base.py"}:
            continue
        module_name = _module_name(path, root)
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            discovered.append(
                {
                    "class_name": path.stem,
                    "class_path": "",
                    "module": module_name,
                    "import_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if predicate(cls):
                discovered.append(
                    {
                        "class_name": cls.__name__,
                        "class_path": f"{module_name}.{cls.__name__}",
                        "module": module_name,
                        "import_error": "",
                    }
                )
    return discovered


def slug(name: str) -> str:
    out = []
    prev_lower = False
    for ch in name:
        if ch.isupper() and prev_lower:
            out.append("_")
        out.append(ch.lower())
        prev_lower = ch.islower() or ch.isdigit()
    return "".join(out).replace("__", "_").strip("_")


def build_pair_config(
    est: dict[str, str],
    ctrl: dict[str, str],
    target_vol: float = 0.10,
    cost_bps: float = 5.0,
    weight_max: float = 1.5,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    est_name = est["class_name"]
    ctrl_name = ctrl["class_name"]
    return {
        "name": f"{slug(est_name)}__{slug(ctrl_name)}",
        "description": "Auto-discovered estimator/controller pair backtest",
        "data": {
            "symbol": "SPY",
            "source": "databento",
            "price_col": "close",
            **({"start_date": start_date} if start_date else {}),
            **({"end_date": end_date} if end_date else {}),
        },
        "target_vol": float(target_vol),
        "rebalance_freq": "daily",
        "roll_window": 252,
        "weight_min": 0.0,
        "weight_max": float(weight_max),
        "cost_bps": float(cost_bps),
        "vol_ann": 252,
        "router": {"enabled": False},
        "estimator": {
            "class": est["class_path"],
            "params": dict(ESTIMATOR_PARAM_DEFAULTS.get(est_name, {"vol_ann": 252})),
        },
        "controller": {
            "class": ctrl["class_path"],
            "params": dict(CONTROLLER_PARAM_DEFAULTS.get(ctrl_name, {})),
        },
    }


def _run_pair_child(cfg: dict, path: str, log_path: str, mode: str, queue: mp.Queue) -> None:
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
        queue.put(
            {
                "status": "error",
                "rows": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def run_pair_with_timeout(cfg: dict, path: Path, log_path: Path, mode: str, timeout_sec: int) -> dict:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_run_pair_child, args=(cfg, str(path), str(log_path), mode, queue))
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        msg = f"Timed out after {timeout_sec} seconds"
        log_path.write_text(msg + "\n", encoding="utf-8")
        return {"status": "timeout", "rows": 0, "error": msg}

    if queue.empty():
        msg = f"Process exited with code {proc.exitcode} without returning a result"
        log_path.write_text(msg + "\n", encoding="utf-8")
        return {"status": "error", "rows": 0, "error": msg}

    out = queue.get()
    return out


def run_all_pairs(
    out_dir: str | Path,
    mode: str = "all",
    timeout_sec: int = 90,
    resume: bool = True,
    target_vol: float = 0.10,
    cost_bps: float = 5.0,
    weight_max: float = 1.5,
    include_pairs: set[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    output_dir = Path(out_dir)
    if not output_dir.is_absolute():
        output_dir = Env.path("root") / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    estimators = discover_classes(Env.path("root") / "src" / "estimators", _is_estimator_class)
    controllers = discover_classes(Env.path("root") / "src" / "controllers", _is_controller_class)

    valid_estimators = [x for x in estimators if x.get("class_path")]
    valid_controllers = [x for x in controllers if x.get("class_path")]

    discovery = pd.DataFrame(
        [{**x, "kind": "estimator"} for x in estimators]
        + [{**x, "kind": "controller"} for x in controllers]
    )
    discovery.to_csv(output_dir / "discovery.csv", index=False)

    records: list[dict[str, Any]] = []
    done_names: set[str] = set()
    manifest_path = output_dir / "manifest.csv"
    if resume and manifest_path.exists():
        previous = pd.read_csv(manifest_path)
        for row in previous.to_dict("records"):
            if row.get("status") == "ok" and Path(str(row.get("path", ""))).exists():
                records.append(row)
                done_names.add(str(row.get("name")))

    total = len(valid_estimators) * len(valid_controllers)
    count = 0

    for est in valid_estimators:
        for ctrl in valid_controllers:
            count += 1
            cfg = build_pair_config(
                est,
                ctrl,
                target_vol=target_vol,
                cost_bps=cost_bps,
                weight_max=weight_max,
                start_date=start_date,
                end_date=end_date,
            )
            name = cfg["name"]
            if include_pairs is not None and name not in include_pairs:
                continue
            path = output_dir / f"{name}.parquet"
            log_path = log_dir / f"{name}.log"

            if name in done_names:
                continue
            if resume and path.exists():
                record = {
                    "name": name,
                    "status": "ok",
                    "path": str(path),
                    "estimator": est["class_path"],
                    "controller": ctrl["class_path"],
                    "rows": int(len(pd.read_parquet(path))),
                    "log": str(log_path),
                }
                records.append(record)
                pd.DataFrame(records).to_csv(manifest_path, index=False)
                continue

            print(f"[{count}/{total}] {name}")
            record: dict[str, Any] = {
                "name": name,
                "status": "ok",
                "path": str(path),
                "estimator": est["class_path"],
                "controller": ctrl["class_path"],
                "rows": 0,
                "log": str(log_path),
            }

            result = run_pair_with_timeout(cfg, path, log_path, mode=mode, timeout_sec=timeout_sec)
            record.update(result)
            if record["status"] != "ok":
                print(f"  failed: {record['error']}")

            records.append(record)
            pd.DataFrame(records).to_csv(manifest_path, index=False)

    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    print(f"\nSaved manifest to {manifest_path}")
    print(f"Successful parquet files: {int((manifest['status'] == 'ok').sum())}")
    print(f"Failed pairs: {int((manifest['status'] != 'ok').sum())}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every discovered estimator/controller pair.")
    parser.add_argument(
        "--out-dir",
        default="results/all_estimator_controller_pairs",
        help="Independent output directory for pair parquet files, logs, and manifest.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "in_sample", "out_of_sample"],
        help="Data slice mode passed to VolTargetEngine.run().",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=90,
        help="Maximum seconds allowed for each individual estimator/controller pair.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip successful pairs already present in manifest.csv.",
    )
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--weight-max", type=float, default=1.5)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument(
        "--include-pairs-file",
        default="",
        help="Optional newline-delimited exact pair names to generate.",
    )
    args = parser.parse_args()
    include_pairs = None
    if args.include_pairs_file:
        include_pairs = {
            line.strip()
            for line in Path(args.include_pairs_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    run_all_pairs(
        args.out_dir,
        mode=args.mode,
        timeout_sec=args.timeout_sec,
        resume=not args.no_resume,
        target_vol=args.target_vol,
        cost_bps=args.cost_bps,
        weight_max=args.weight_max,
        include_pairs=include_pairs,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
    )


if __name__ == "__main__":
    main()
