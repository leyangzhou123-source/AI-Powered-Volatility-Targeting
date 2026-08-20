"""Run backtests for every estimator/controller combination.

This script discovers estimator/controller classes from ``src/``, reuses
existing strategy YAMLs as parameter templates when available, and runs the
backtest engine once per Cartesian-product pair.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import yaml

from src.backtest import VolTargetEngine
from src.env import Env


ROOT = Path(__file__).parents[1]
SRC_DIR = ROOT / "src"
STRATEGY_DIR = Env.path("strategies")
RESULTS_DIR = Env.path("results")

EXCLUDED_ESTIMATORS = {"base"}
EXCLUDED_CONTROLLERS = {"__init__"}


def snake_case(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("-", "_").lower()


def discover_classes(package_dir: Path, package_root: str, excluded: set[str]) -> dict[str, str]:
    classes: dict[str, str] = {}
    for path in sorted(package_dir.glob("*.py")):
        if path.stem in excluded:
            continue

        text = path.read_text(encoding="utf-8")
        class_names = re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|:)", text, flags=re.M)
        if not class_names:
            continue

        class_name = class_names[0]
        module_path = f"{package_root}.{path.stem}"
        classes[class_name] = f"{module_path}.{class_name}"
    return classes


def canonical_class_id(class_path: str | None) -> str | None:
    if not class_path:
        return None
    return class_path.rsplit(".", 1)[-1].strip().lower()


def load_strategy_templates() -> tuple[dict[str, dict], dict[str, dict]]:
    estimator_templates: dict[str, dict] = {}
    controller_templates: dict[str, dict] = {}

    for cfg_path in sorted(STRATEGY_DIR.glob("*.yaml")):
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        est_cfg = deepcopy(cfg.get("estimator", {}))
        ctrl_cfg = deepcopy(cfg.get("controller", {}))

        est_key = canonical_class_id(est_cfg.get("class"))
        ctrl_key = canonical_class_id(ctrl_cfg.get("class"))

        if est_key and est_key not in estimator_templates:
            estimator_templates[est_key] = {
                "source_config": str(cfg_path),
                "top_level": {
                    k: deepcopy(v)
                    for k, v in cfg.items()
                    if k not in {"name", "description", "controller"}
                },
                "estimator": est_cfg,
            }

        if ctrl_key and ctrl_key not in controller_templates:
            controller_templates[ctrl_key] = {
                "source_config": str(cfg_path),
                "controller": ctrl_cfg,
            }

    return estimator_templates, controller_templates


def build_config(
    estimator_class: str,
    controller_class: str,
    estimator_templates: dict[str, dict],
    controller_templates: dict[str, dict],
) -> dict:
    est_key = canonical_class_id(estimator_class)
    ctrl_key = canonical_class_id(controller_class)

    est_template = deepcopy(estimator_templates.get(est_key, {}))
    ctrl_template = deepcopy(controller_templates.get(ctrl_key, {}))

    cfg = deepcopy(est_template.get("top_level", {}))
    cfg.setdefault("data", {"path": "data/processed/ES_Daily_Processed.parquet"})
    cfg.setdefault("roll_window", 252)
    cfg.setdefault("target_vol", 0.10)
    cfg.setdefault("rebalance_freq", "D")
    cfg.setdefault("weight_min", 0.0)
    cfg.setdefault("weight_max", 1.5)
    cfg.setdefault("cost_bps", 5.0)
    cfg.setdefault("initial_capital", 1000.0)

    cfg["estimator"] = deepcopy(est_template.get("estimator", {}))
    cfg["estimator"]["class"] = estimator_class
    cfg["estimator"].setdefault("params", {})

    cfg["controller"] = deepcopy(ctrl_template.get("controller", {}))
    cfg["controller"]["class"] = controller_class
    cfg["controller"].setdefault("params", {})

    est_name = snake_case(estimator_class.rsplit(".", 1)[-1])
    ctrl_name = snake_case(controller_class.rsplit(".", 1)[-1])
    cfg["name"] = f"{est_name}__{ctrl_name}"

    return cfg


def run_combinations(
    estimator_filter: set[str] | None = None,
    controller_filter: set[str] | None = None,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> list[dict]:
    estimators = discover_classes(SRC_DIR / "estimators", "src.estimators", EXCLUDED_ESTIMATORS)
    controllers = discover_classes(SRC_DIR / "controllers", "src.controllers", EXCLUDED_CONTROLLERS)

    estimator_templates, controller_templates = load_strategy_templates()

    estimator_items = sorted(estimators.items())
    controller_items = sorted(controllers.items())

    if estimator_filter:
        estimator_items = [item for item in estimator_items if item[0] in estimator_filter]
    if controller_filter:
        controller_items = [item for item in controller_items if item[0] in controller_filter]

    records: list[dict] = []
    run_count = 0

    for est_name, est_class in estimator_items:
        for ctrl_name, ctrl_class in controller_items:
            cfg = build_config(est_class, ctrl_class, estimator_templates, controller_templates)
            record = {
                "strategy": cfg["name"],
                "estimator": est_name,
                "controller": ctrl_name,
                "status": "pending",
                "result_path": str(RESULTS_DIR / f"{cfg['name']}.parquet"),
                "error": "",
            }

            print(f"\n{'=' * 72}")
            print(f"{cfg['name']}: {est_class} x {ctrl_class}")
            print(f"{'=' * 72}")

            if dry_run:
                record["status"] = "dry_run"
                records.append(record)
            else:
                try:
                    engine = VolTargetEngine.from_config(cfg)
                    engine.run()
                    out_path = engine.save()
                    engine.summary()
                    record["status"] = "ok"
                    record["result_path"] = str(out_path)
                except Exception as exc:
                    record["status"] = "error"
                    record["error"] = str(exc)
                    print(f"FAILED: {exc}")
                records.append(record)

            run_count += 1
            if max_runs is not None and run_count >= max_runs:
                return records

    return records


def write_summary(records: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "all_combinations_summary.csv"
    fieldnames = ["strategy", "estimator", "controller", "status", "result_path", "error"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return out_path


def parse_name_list(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {chunk.strip() for chunk in raw.split(",") if chunk.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run backtests for every estimator/controller combination."
    )
    parser.add_argument(
        "--estimators",
        help="Comma-separated estimator class names to include, e.g. AR1,EWMA,HARRV",
    )
    parser.add_argument(
        "--controllers",
        help="Comma-separated controller class names to include, e.g. NaiveScaling,RegimeSwitchController",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the combinations and write a summary without running the engine.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Stop after N combinations. Helpful for smoke tests.",
    )
    args = parser.parse_args()

    records = run_combinations(
        estimator_filter=parse_name_list(args.estimators),
        controller_filter=parse_name_list(args.controllers),
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    summary_path = write_summary(records)

    ok = sum(r["status"] == "ok" for r in records)
    dry = sum(r["status"] == "dry_run" for r in records)
    err = sum(r["status"] == "error" for r in records)

    print(f"\nSummary written to: {summary_path}")
    print(f"Total combinations processed: {len(records)}")
    print(f"Successful: {ok}")
    print(f"Dry-run only: {dry}")
    print(f"Errors: {err}")


if __name__ == "__main__":
    main()
