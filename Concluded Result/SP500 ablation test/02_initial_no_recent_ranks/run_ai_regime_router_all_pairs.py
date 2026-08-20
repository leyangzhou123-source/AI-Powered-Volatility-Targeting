"""Run the all-pairs router with AI-generated regime context."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.evaluate_router_protocol_precomputed import (  # noqa: E402
    evaluate_precomputed_router_protocol,
)


def load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _as_lower_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).lower() for item in value]
    return [str(value).lower()]


def _pair_cfg_is_excluded(pair_cfg: dict, params: dict) -> bool:
    excluded_estimators = _as_lower_list(params.get("excluded_estimators", []))
    excluded_controllers = _as_lower_list(params.get("excluded_controllers", []))
    excluded_pairs_containing = _as_lower_list(params.get("excluded_pairs_containing", []))
    estimator = pair_cfg.get("estimator", {}) or {}
    controller = pair_cfg.get("controller", {}) or {}
    estimator_text = " ".join(
        [
            str(estimator.get("class", "")) if isinstance(estimator, dict) else str(estimator),
            str(estimator.get("name", "")) if isinstance(estimator, dict) else "",
        ]
    ).lower()
    controller_text = " ".join(
        [
            str(controller.get("class", "")) if isinstance(controller, dict) else str(controller),
            str(controller.get("name", "")) if isinstance(controller, dict) else "",
        ]
    ).lower()
    pair_text = " ".join([str(pair_cfg.get("name", "")), estimator_text, controller_text]).lower()
    if any(token in estimator_text for token in excluded_estimators):
        return True
    if any(token in controller_text for token in excluded_controllers):
        return True
    if any(token in pair_text for token in excluded_pairs_containing):
        return True
    return False


def _load_pair_pool_from_controller_diagnostics(path: str) -> set[str]:
    pool_path = Path(path)
    if not pool_path.exists():
        raise FileNotFoundError(f"Pair pool diagnostics file not found: {path}")
    names: set[str] = set()
    with pool_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("window", "")).lower() == "train":
                name = str(row.get("pair", "")).strip()
                if name:
                    names.add(name)
    if not names:
        raise ValueError(f"No train-window pairs found in pair pool file: {path}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI-regime all-pairs router backtest.")
    parser.add_argument("--strategy", default="configs/strategies/router_master.yaml")
    parser.add_argument("--name", default="router_master_ai_regime_no_ar_rf_oos")
    parser.add_argument("--mode", default="out_of_sample", choices=["all", "in_sample", "out_of_sample"])
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--max-ai-regime-calls", type=int, default=365)
    parser.add_argument("--max-ai-selection-calls", type=int, default=365)
    parser.add_argument("--max-total-ai-calls", type=int, default=365)
    parser.add_argument("--ai-regime-interval", type=int, default=10)
    parser.add_argument("--ai-selection-interval", type=int, default=10)
    parser.add_argument("--candidate-top-n", type=int, default=6)
    parser.add_argument(
        "--router-ablation",
        choices=[
            "none",
            "no_regime",
            "no_recent_ranks",
            "no_benchmark_baseline",
            "deterministic_switch",
            "deterministic_selection",
        ],
        default="none",
    )
    parser.add_argument("--precomputed-ai-regime-path", default="results/evaluation/ai_regime_series/ai_regime_10d.csv")
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--request-min-interval-seconds", type=float, default=None)
    parser.add_argument("--weight-max", type=float, default=None)
    parser.add_argument("--pair-results-dir", default="results/all_estimator_controller_pairs")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--step-days", type=int, default=252)
    parser.add_argument("--oos-start-date", default="2023-01-01")
    parser.add_argument("--oos-end-date", default=None)
    parser.add_argument("--single-oos-window", action="store_true", default=True)
    parser.add_argument("--freeze-oos-metrics", action="store_true", default=False)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-dir", default="results/evaluation/ai_regime_router_precomputed")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--extra-env-file", default="tests/.env")
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Optional environment variable name to use as the AI API key, e.g. NVAPI_KEY2.",
    )
    parser.add_argument(
        "--exclude-pairs",
        default="",
        help="Comma-separated exact router pair names to remove before evaluation.",
    )
    parser.add_argument(
        "--pair-pool-controller-diagnostics",
        default="",
        help="Controller diagnostics CSV whose train-window pair list defines the exact eligible pool.",
    )
    parser.add_argument(
        "--skip-router-param-exclusions",
        action="store_true",
        default=False,
        help="Do not apply params excluded_estimators/excluded_controllers filtering.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    load_env_file(args.extra_env_file)

    if args.precomputed_ai_regime_path:
        regime_path = Path(args.precomputed_ai_regime_path)
        if not regime_path.exists():
            raise FileNotFoundError(
                f"Precomputed AI regime series not found: {args.precomputed_ai_regime_path}"
            )

    with open(args.strategy, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.pair_pool_controller_diagnostics:
        exact_pool_names = _load_pair_pool_from_controller_diagnostics(
            args.pair_pool_controller_diagnostics
        )
        pairs = list((cfg.get("router", {}) or {}).get("pairs", []) or [])
        kept_pairs = [
            pair for pair in pairs if str(pair.get("name", "")) in exact_pool_names
        ]
        missing = sorted(
            exact_pool_names - {str(pair.get("name", "")) for pair in kept_pairs}
        )
        if missing:
            raise ValueError(
                "Pair pool file contains names not present in strategy config: "
                + ", ".join(missing[:20])
            )
        cfg.setdefault("router", {})["pairs"] = kept_pairs
        print(
            "[ai-regime-run] "
            f"loaded_exact_pair_pool={len(kept_pairs)} "
            f"pool_file={args.pair_pool_controller_diagnostics}",
            flush=True,
        )

    excluded_pair_names = {
        item.strip()
        for item in str(args.exclude_pairs or "").split(",")
        if item.strip()
    }
    if excluded_pair_names:
        pairs = list((cfg.get("router", {}) or {}).get("pairs", []) or [])
        kept_pairs = [
            pair for pair in pairs if str(pair.get("name", "")) not in excluded_pair_names
        ]
        cfg.setdefault("router", {})["pairs"] = kept_pairs
        removed = len(pairs) - len(kept_pairs)
        print(
            "[ai-regime-run] "
            f"removed_exact_pairs={removed} "
            f"requested_exact_pair_exclusions={sorted(excluded_pair_names)}",
            flush=True,
        )

    cfg["name"] = args.name
    cfg["estimator"] = {
        "class": "src.estimators.ewma.EWMA",
        "params": {"halflife": 20, "vol_ann": 252},
    }
    cfg["controller"] = {
        "class": "src.controllers.naive_scaling.NaiveScaling",
        "params": {},
    }
    if args.weight_max is not None:
        cfg["weight_max"] = float(args.weight_max)

    cfg["intraday_realized_vol"] = {
        "enabled": True,
        "path": "data/processed/SP500_Intraday_RealizedVol.parquet",
        "lookback": 21,
    }
    cfg["pair_history_features"] = {
        "enabled": True,
        "path": "results/all_estimator_controller_pairs",
        "benchmark_path": "results/realized_vol_22__naive_scaling_oos.parquet",
        "lookbacks": [10, 63, 126],
    }

    router_cfg = cfg.setdefault("router", {})
    router_cfg["enabled"] = True
    router_cfg["type"] = "ai_regime"
    router_cfg["class"] = "src.router.ai_regime_router.AIRegimeRouter"
    params = router_cfg.setdefault("params", {})
    explicit_api_key = os.getenv(args.api_key_env, "") if args.api_key_env else ""
    params.update(
        {
            "provider": args.provider,
            "api_format": "chat_completions",
            "model": args.model,
            "api_key": explicit_api_key
            or os.getenv("NVAPI_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY", ""),
            "response_format": {"type": "json_object"},
            "reasoning_effort": None,
            "max_ai_regime_calls": args.max_ai_regime_calls,
            "max_ai_selection_calls": args.max_ai_selection_calls,
            "max_total_ai_calls": args.max_total_ai_calls,
            "ai_regime_interval": args.ai_regime_interval,
            "ai_regime_on_event": False,
            "ai_selection_interval": args.ai_selection_interval,
            "candidate_top_n": args.candidate_top_n,
            "precomputed_ai_regime_path": args.precomputed_ai_regime_path,
            "precomputed_ai_regime_only": bool(args.precomputed_ai_regime_path),
            "raw_response_debug_dir": "results/evaluation/ai_regime_raw_responses",
            "regime_suitability_path": "results/evaluation/regime_suitability_all_pairs",
            "use_heuristic_regime_bias": False,
            "regime_bias_weight": 3.0,
            "regime_suitability_scale": 1.0,
            "sticky_period": 1,
            "perf_weight": 1.5,
            "lambda_drawdown": 0.5,
            "lambda_switch": 0.15,
            "lambda_invalid": 2.0,
            "lambda_exception": 1.0,
            "excluded_estimators": [
                "ar1",
                "ar2",
                "ewma",
                "hybrid_ewma",
                "hybrid_ewmaregime",
                "lasso",
                "lasso_volatility",
                "buy_and_hold",
                "buyandhold",
            ],
            "excluded_controllers": [
                "constant",
                "constant_weight",
                "cva_restargeting",
                "cvar",
                "cvar_es",
                "cvares",
            ],
            "fail_open": True,
            "retry_ai_after_failure": False,
            "max_output_tokens": 128,
            "switch_decision_max_output_tokens": 64,
            "selection_max_output_tokens": 96,
            "temperature": 0.0,
            "timeout": args.timeout,
            "request_min_interval_seconds": (
                float(args.request_min_interval_seconds)
                if args.request_min_interval_seconds is not None
                else (2.0 if str(args.provider).lower() == "nvidia" else 0.0)
            ),
        }
    )

    ablation = str(args.router_ablation)
    params["disable_regime_context"] = ablation == "no_regime"
    params["disable_recent_rank_context"] = ablation == "no_recent_ranks"
    params["disable_benchmark_context"] = ablation == "no_benchmark_baseline"
    params["disable_deterministic_baseline_context"] = ablation == "no_benchmark_baseline"
    params["deterministic_switch_decision"] = ablation == "deterministic_switch"
    params["deterministic_pair_selection"] = ablation == "deterministic_selection"

    pairs_before_router_filter = list((cfg.get("router", {}) or {}).get("pairs", []) or [])
    if args.skip_router_param_exclusions:
        hard_filtered_pairs = pairs_before_router_filter
        router_param_removed = 0
        params["excluded_estimators"] = []
        params["excluded_controllers"] = []
        params["excluded_pairs_containing"] = []
        print(
            "[ai-regime-run] skipped_router_param_exclusions=True",
            flush=True,
        )
    else:
        hard_filtered_pairs = [
            pair for pair in pairs_before_router_filter if not _pair_cfg_is_excluded(pair, params)
        ]
        cfg.setdefault("router", {})["pairs"] = hard_filtered_pairs
        router_param_removed = len(pairs_before_router_filter) - len(hard_filtered_pairs)
        if router_param_removed:
            print(
                "[ai-regime-run] "
                f"hard_removed_router_param_exclusions={router_param_removed}",
                flush=True,
            )

    print(
        "[ai-regime-run] "
        f"model={args.model} provider={args.provider} "
        f"router_ablation={args.router_ablation} "
        f"precomputed_ai_regime_path={args.precomputed_ai_regime_path or 'NONE'} "
        f"precomputed_ai_regime_only={bool(args.precomputed_ai_regime_path)} "
        f"ai_regime_interval={args.ai_regime_interval} "
        f"ai_selection_interval={args.ai_selection_interval} "
        f"candidate_top_n={args.candidate_top_n} "
        f"excluded_estimators={params['excluded_estimators']} "
        f"excluded_controllers={params['excluded_controllers']}",
        flush=True,
    )

    tables = evaluate_precomputed_router_protocol(
        config_path=args.strategy,
        pair_results_dir=Path(args.pair_results_dir),
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        max_pairs=args.max_pairs,
        output_dir=Path(args.output_dir),
        config=cfg,
        oos_start_date=args.oos_start_date,
        oos_end_date=args.oos_end_date,
        single_oos_window=args.single_oos_window,
        freeze_oos_metrics=args.freeze_oos_metrics,
    )
    payload = {
        "output_dir": args.output_dir,
        "summary": tables["summary"].to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
