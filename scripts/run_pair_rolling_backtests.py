"""Run rolling-window backtests for every estimator/controller pair in a config."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.backtest import VolTargetEngine
from src.env import Env


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        path = Env.path("strategies") / path.name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"YAML loaded as None: {path}")
    return cfg


def iter_pair_configs(master_cfg: dict):
    router_cfg = master_cfg.get("router", {}) or {}
    pairs = router_cfg.get("pairs", []) or []

    if pairs:
        for pair in pairs:
            cfg = copy.deepcopy(master_cfg)
            cfg["name"] = str(pair["name"])
            cfg["estimator"] = copy.deepcopy(pair["estimator"])
            cfg["controller"] = copy.deepcopy(pair["controller"])
            cfg["router"] = {"enabled": False}
            yield cfg
        return

    if "estimator" in master_cfg and "controller" in master_cfg:
        cfg = copy.deepcopy(master_cfg)
        cfg["router"] = {"enabled": False}
        yield cfg
        return

    raise ValueError("Config must contain router.pairs or a top-level estimator/controller.")


def run_pair_backtests(config_path: str | Path, out_dir: str | Path, mode: str = "all") -> pd.DataFrame:
    master_cfg = load_config(config_path)
    output_dir = Path(out_dir)
    if not output_dir.is_absolute():
        output_dir = Env.path("root") / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for pair_cfg in iter_pair_configs(master_cfg):
        name = pair_cfg["name"]
        print(f"\nRunning rolling pair: {name}")
        record = {"name": name, "status": "ok", "path": str(output_dir / f"{name}.parquet")}

        try:
            engine = VolTargetEngine.from_config(pair_cfg)
            result = engine.run(mode=mode)
            result.to_parquet(output_dir / f"{name}.parquet")
            record["rows"] = int(len(result))
            record["estimator"] = pair_cfg["estimator"]["class"]
            record["controller"] = pair_cfg["controller"]["class"]
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"Failed {name}: {record['error']}")

        records.append(record)

    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    ok = int((manifest["status"] == "ok").sum()) if not manifest.empty else 0
    failed = int((manifest["status"] != "ok").sum()) if not manifest.empty else 0
    print(f"\nSaved {ok} pair parquet files to {output_dir}")
    if failed:
        print(f"{failed} pairs failed; see manifest.csv")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rolling-window volatility/weight backtests for each estimator/controller pair."
    )
    parser.add_argument(
        "--strategy",
        default="router_master.yaml",
        help="Config path or filename under configs/strategies.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/rolling_pairs",
        help="Directory where pair parquet files and manifest.csv are written.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "in_sample", "out_of_sample"],
        help="Data slice mode passed to VolTargetEngine.run().",
    )
    args = parser.parse_args()

    run_pair_backtests(args.strategy, args.out_dir, mode=args.mode)


if __name__ == "__main__":
    main()
