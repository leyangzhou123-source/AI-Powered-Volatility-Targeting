"""Deterministically tune non-AI router parameters.

Stage 1 LLM event timing is intentionally excluded. This script tunes ordinary
router knobs with a small walk-forward grid over precomputed pair backtests.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.evaluate_router_protocol_precomputed import evaluate_precomputed_router_protocol  # noqa: E402
from src.env import Env  # noqa: E402


EXCLUDED_AI_KEYS = {
    "api_key",
    "api_format",
    "endpoint",
    "model",
    "provider",
    "decision_mode",
    "decision_interval",
    "always_call_first",
    "min_decision_gap",
    "rv_zscore_trigger",
    "rv_change_trigger",
    "rv_percentile_trigger",
    "max_calls",
    "max_output_tokens",
    "temperature",
    "timeout",
}


def _parse_values(raw: str) -> list[Any]:
    values: list[Any] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        low = token.lower()
        if low in ("true", "false"):
            values.append(low == "true")
            continue
        try:
            if any(ch in token for ch in ".eE"):
                values.append(float(token))
            else:
                values.append(int(token))
        except ValueError:
            values.append(token)
    return values


def _default_grid(quick: bool) -> dict[str, list[Any]]:
    if quick:
        return {
            "sticky_period": [1, 5],
            "perf_weight": [1.0, 1.5],
            "lambda_drawdown": [0.25, 0.5],
            "lambda_switch": [0.05, 0.15],
            "regime_suitability_scale": [0.5, 0.75],
        }
    return {
        "sticky_period": [1, 3, 5, 10],
        "perf_weight": [1.0, 1.5, 2.0],
        "lambda_drawdown": [0.15, 0.25, 0.5, 0.75],
        "lambda_switch": [0.0, 0.05, 0.15, 0.25],
        "regime_suitability_scale": [0.25, 0.5, 0.75, 1.0],
    }


def _grid_from_args(args: argparse.Namespace) -> dict[str, list[Any]]:
    grid = _default_grid(args.quick)
    for spec in args.param:
        if "=" not in spec:
            raise ValueError(f"Parameter grid entry must be name=v1,v2,... got {spec!r}")
        key, raw_values = spec.split("=", 1)
        key = key.strip()
        if key in EXCLUDED_AI_KEYS:
            raise ValueError(f"{key!r} is an AI/event/API key; leave Stage 1 calibration to the AI script.")
        values = _parse_values(raw_values)
        if not values:
            raise ValueError(f"No values supplied for {key!r}")
        grid[key] = values
    return grid


def _param_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _objective(summary: pd.DataFrame, router: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    sharpe = _safe_float(row.get("overall_sharpe"))
    ann_return = _safe_float(row.get("overall_ann_return"))
    drawdown = _safe_float(row.get("overall_drawdown"))
    cvar = _safe_float(row.get("overall_cvar_95"))
    ann_vol = _safe_float(row.get("overall_ann_vol"))
    avg_switch_rate = _safe_float(router["switch_rate"].mean()) if "switch_rate" in router.columns else 0.0
    avg_turnover = _safe_float(router["turnover"].mean()) if "turnover" in router.columns else 0.0
    avg_unique = _safe_float(router["unique_selected"].mean()) if "unique_selected" in router.columns else 0.0
    below_vol = max(0.0, args.vol_min - ann_vol)
    above_vol = max(0.0, ann_vol - args.vol_max)
    vol_band_penalty = below_vol + above_vol
    score = (
        args.return_weight * ann_return
        + args.sharpe_weight * sharpe
        - args.drawdown_penalty * drawdown
        - args.cvar_penalty * cvar
        - args.turnover_penalty * avg_turnover
        - args.switch_penalty * avg_switch_rate
        - args.vol_penalty * vol_band_penalty
        + args.diversity_bonus * np.log1p(avg_unique)
    )
    return {
        "objective": float(score),
        "overall_ann_return": ann_return,
        "overall_sharpe": sharpe,
        "overall_drawdown": drawdown,
        "overall_cvar_95": cvar,
        "overall_ann_vol": ann_vol,
        "vol_band_penalty": vol_band_penalty,
        "avg_turnover": avg_turnover,
        "avg_switch_rate": avg_switch_rate,
        "avg_unique_selected": avg_unique,
    }


def _yaml_snippet(best_params: dict[str, Any]) -> dict[str, Any]:
    return {"router": {"params": best_params}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune non-AI router parameters with precomputed walk-forward backtests.")
    parser.add_argument("--strategy", "-s", default="configs/strategies/router_master.yaml")
    parser.add_argument("--pair-results-dir", default="results/all_estimator_controller_pairs")
    parser.add_argument("--output-dir", default=str(Env.path("evaluation") / "router_param_tuning"))
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Use a tiny grid for smoke testing.")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override/add grid entry, e.g. --param sticky_period=5,10,21. AI/event/API keys are rejected.",
    )
    parser.add_argument("--return-weight", type=float, default=4.0)
    parser.add_argument("--sharpe-weight", type=float, default=0.25)
    parser.add_argument("--drawdown-penalty", type=float, default=1.0)
    parser.add_argument("--cvar-penalty", type=float, default=0.50)
    parser.add_argument("--turnover-penalty", type=float, default=0.25)
    parser.add_argument("--switch-penalty", type=float, default=0.15)
    parser.add_argument("--vol-penalty", type=float, default=8.0)
    parser.add_argument("--vol-min", type=float, default=0.09)
    parser.add_argument("--vol-max", type=float, default=0.11)
    parser.add_argument("--diversity-bonus", type=float, default=0.02)
    parser.add_argument("--target-vol", type=float, default=0.10, help="Kept for compatibility; use --vol-min/--vol-max for scoring.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.strategy, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    grid = _grid_from_args(args)
    combos = _param_combos(grid)
    rows: list[dict[str, Any]] = []

    print(f"Tuning {len(combos)} non-AI parameter combinations")
    for idx, combo in enumerate(combos, start=1):
        cfg = copy.deepcopy(base_cfg)
        params = cfg.setdefault("router", {}).setdefault("params", {})
        params.update(combo)
        run_dir = out_dir / f"run_{idx:03d}"
        tables = evaluate_precomputed_router_protocol(
            config_path=args.strategy,
            pair_results_dir=Path(args.pair_results_dir),
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            max_pairs=args.max_pairs,
            output_dir=run_dir,
            config=cfg,
            verbose=False,
        )
        metrics = _objective(tables["summary"], tables["router"], args)
        rows.append({"run": idx, **combo, **metrics})
        print(
            f"[{idx:03d}/{len(combos):03d}] objective={metrics['objective']:.4f} "
            f"ann_ret={metrics['overall_ann_return']:.3f} vol={metrics['overall_ann_vol']:.3f} "
            f"dd={metrics['overall_drawdown']:.3f} turnover={metrics['avg_turnover']:.3f} "
            f"params={json.dumps(combo, sort_keys=True)}"
        )

    results = pd.DataFrame(rows).sort_values("objective", ascending=False)
    results.to_csv(out_dir / "router_param_tuning_results.csv", index=False)

    best = results.iloc[0].to_dict()
    best_params = {key: best[key] for key in grid if key in best}
    for key, values in grid.items():
        if values and isinstance(values[0], int) and not isinstance(values[0], bool):
            best_params[key] = int(best_params[key])
        elif values and isinstance(values[0], float):
            best_params[key] = float(best_params[key])

    with open(out_dir / "best_router_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_yaml_snippet(best_params), f, sort_keys=False)

    print("=" * 72)
    print("Best non-AI router params")
    print(yaml.safe_dump(_yaml_snippet(best_params), sort_keys=False).strip())
    print(results.head(10).to_string(index=False))
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
