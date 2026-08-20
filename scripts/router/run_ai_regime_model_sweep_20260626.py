from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.evaluate_router_protocol_precomputed import evaluate_precomputed_router_protocol


def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def base_config(model: str, run_name: str, raw_dir: str) -> dict:
    with open("configs/strategies/router_master.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    api_key = os.getenv("NVAPI_KEY") or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing NVAPI_KEY/NVIDIA_API_KEY/OPENAI_API_KEY")

    cfg["name"] = run_name
    cfg["estimator"] = {
        "class": "src.estimators.ewma.EWMA",
        "params": {"halflife": 20, "vol_ann": 252},
    }
    cfg["controller"] = {"class": "src.controllers.naive_scaling.NaiveScaling", "params": {}}
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
    params.update(
        {
            "provider": "nvidia",
            "api_format": "chat_completions",
            "model": model,
            "api_key": api_key,
            "response_format": None,
            "reasoning_effort": None,
            "max_ai_regime_calls": 365,
            "max_ai_selection_calls": 365,
            "max_total_ai_calls": 365,
            "ai_regime_interval": 10,
            "ai_regime_on_event": False,
            "ai_selection_interval": 10,
            "candidate_top_n": 6,
            "precomputed_ai_regime_path": "results/evaluation/ai_regime_series/ai_regime_10d.csv",
            "precomputed_ai_regime_only": True,
            "raw_response_debug_dir": raw_dir,
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
                "random_forest_vol_estimator",
                "buy_and_hold",
                "buyandhold",
                "ewma",
            ],
            "excluded_controllers": ["constant", "constant_weight", "cvar", "cvares"],
            "fail_open": True,
            "retry_ai_after_failure": False,
            "max_output_tokens": 256,
            "switch_decision_max_output_tokens": 256,
            "selection_max_output_tokens": 512,
            "temperature": 0.0,
            "timeout": 25.0,
        }
    )
    return cfg


def run_one(label: str, model: str) -> dict:
    safe_label = label.lower().replace(" ", "_").replace("-", "_")
    out = Path(f"results/evaluation/ai_regime_router_2023_to_end_no_ewma_10d_{safe_label}_20260626")
    cfg = base_config(
        model=model,
        run_name=f"router_master_ai_regime_10d_{safe_label}",
        raw_dir=f"results/evaluation/ai_regime_raw_responses_10d_{safe_label}_20260626",
    )
    tables = evaluate_precomputed_router_protocol(
        config_path="configs/strategies/router_master.yaml",
        pair_results_dir=Path("results/all_estimator_controller_pairs"),
        train_days=504,
        test_days=252,
        step_days=252,
        max_pairs=None,
        output_dir=out,
        config=cfg,
        oos_start_date="2023-01-01",
        oos_end_date=None,
        single_oos_window=True,
        freeze_oos_metrics=False,
    )
    return {
        "label": label,
        "model": model,
        "output_dir": str(out),
        "summary": tables["summary"].to_dict(orient="records"),
        "diagnostics": tables["router_oos"].to_dict(orient="records")
        if "router_oos" in tables
        else tables.get("router_oos_diagnostics", tables["router"]).to_dict(orient="records"),
    }


def main() -> None:
    load_env(".env")
    runs = [
        ("llama_8b", "meta/llama-3.1-8b-instruct"),
        ("qwen_3", "qwen/qwen3-next-80b-a3b-instruct"),
        ("deepseek_r1", "deepseek-ai/deepseek-r1"),
    ]
    results = []
    for label, model in runs:
        try:
            results.append(run_one(label, model))
        except Exception as exc:
            results.append({"label": label, "model": model, "error": str(exc)})
    Path("paper result folder").mkdir(exist_ok=True)
    out = Path("paper result folder/ai_regime_10d_model_sweep_20260626.json")
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
