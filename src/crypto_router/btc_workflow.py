"""Run the BTC-only AI switching crypto volatility router."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.crypto_router.workflow import (  # noqa: E402
    BASELINE_PAIR,
    _asset_cfg,
    _pair_cfgs,
    build_pair_results,
    build_router_config,
    load_env_file,
    run_router,
    _redacted_router_config,
    write_baseline_comparison,
)
from src.env import Env  # noqa: E402


def build_btc_guard_config(args, pair_results_dir: Path) -> dict:
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing API key: {args.api_key_env} was not found after loading {args.env_file}")
    cfg = build_router_config(
        asset="btc",
        pair_results_dir=pair_results_dir,
        config_path=args.config_out,
        precomputed_ai_regime_path=args.precomputed_ai_regime_path,
        max_ai_calls=args.max_ai_calls,
        model=args.model,
        provider=args.provider,
        api_key=api_key,
        timeout=args.timeout,
        max_output_tokens=args.max_output_tokens,
        switch_decision_max_output_tokens=args.switch_decision_max_output_tokens,
        selection_max_output_tokens=args.selection_max_output_tokens,
        request_min_interval_seconds=args.request_min_interval_seconds,
        api_key_env=args.api_key_env,
    )
    router = cfg.setdefault("router", {})
    router["class"] = "src.crypto_router.ai_vol_target_router.CryptoVolTargetAIRouter"
    params = router.setdefault("params", {})
    params.update(
        {
            "target_vol": float(cfg.get("target_vol", 0.35)),
            "ann_factor": float(cfg.get("vol_ann", 365)),
            "asset_label": "BTC",
            "train_candidate_filter_enabled": True,
            "train_candidate_pool_size": 30,
            "initial_pair_rule": "train_shape",
            "deterministic_switch_decision": False,
            "deterministic_pair_selection": False,
            "sensitiveness": "high",
            "sticky_period": 20,
            "ai_regime_interval": 20,
            "ai_selection_interval": 20,
            "candidate_top_n": 10,
        }
    )
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        __import__("yaml").safe_dump(_redacted_router_config(cfg, args.api_key_env), sort_keys=False),
        encoding="utf-8",
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BTC benchmark-guarded volatility router.")
    parser.add_argument("--mode", choices=["all", "in_sample", "out_of_sample"], default="all")
    parser.add_argument("--pair-results-dir", default="results/crypto_btc_vol_pairs_oss20b")
    parser.add_argument("--output-dir", default="results/evaluation/crypto_btc_benchmark_guard_router_oss20b")
    parser.add_argument("--config-out", type=Path, default=Env.path("strategies") / "crypto_btc_benchmark_guard_router.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--precomputed-ai-regime-path", default="")
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--api-key-env", default="NVAPI_KEY")
    parser.add_argument("--max-ai-calls", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--switch-decision-max-output-tokens", type=int, default=0)
    parser.add_argument("--selection-max-output-tokens", type=int, default=0)
    parser.add_argument("--request-min-interval-seconds", type=float, default=0.0)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=180)
    parser.add_argument("--step-days", type=int, default=180)
    parser.add_argument("--oos-start-date", default="2023-01-01")
    parser.add_argument("--oos-end-date", default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--freeze-oos-metrics", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    pair_results_dir = Path(args.pair_results_dir)
    output_dir = Path(args.output_dir)

    manifest = build_pair_results("btc", pair_results_dir, mode=args.mode, resume=not args.no_resume)
    cfg = build_btc_guard_config(args, pair_results_dir)
    tables = run_router("btc", pair_results_dir, output_dir, cfg, args)

    baseline_path = pair_results_dir / f"btc_{BASELINE_PAIR['name_suffix']}.parquet"
    comparison_path = write_baseline_comparison(output_dir, baseline_path, ann=float(cfg.get("vol_ann", 365)))
    payload = {
        "asset": "btc",
        "pair_results_dir": str(pair_results_dir),
        "config": str(args.config_out),
        "output_dir": str(output_dir),
        "pairs": int((manifest["status"] == "ok").sum()),
        "routed_pairs": len(_pair_cfgs("btc", _asset_cfg("btc"))),
        "baseline": str(baseline_path),
        "baseline_comparison": str(comparison_path) if comparison_path else "",
        "summary": tables["summary"].to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
