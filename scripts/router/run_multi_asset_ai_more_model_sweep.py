"""Run additional multi-asset AI portfolio router model variants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.run_multi_asset_ai_model_sweep import (  # noqa: E402
    INCLUDED_PAIRS,
    audit_no_leak,
    export_paper_files,
    metrics,
)
from src.router.ai_portfolio_regime_router import run_ai_portfolio_router  # noqa: E402


MODEL_RUNS = [
    ("llama_33_70b", "meta/llama-3.3-70b-instruct"),
    ("llama_31_70b", "meta/llama-3.1-70b-instruct"),
    ("mixtral_8x7b", "mistralai/mixtral-8x7b-instruct-v0.1"),
]


def main() -> None:
    manifest = "results/multi_asset_tuned_pairs_vol10/manifest.csv"
    outdir = Path("results/multi_asset_tuned_pairs_vol10/model_sweep")
    outdir.mkdir(parents=True, exist_ok=True)
    base_params = {
        "api_key_env": "NVAPI_KEY",
        "ai_enabled": True,
        "ai_retries": 3,
        "timeout": 90,
        "max_output_tokens": 1024,
        "reasoning_effort": "low",
        "precomputed_regime_path": "results/multi_asset_tuned_pairs_vol10/ai_volatility_regime_series_macro_inputs_start20230210_interval10_noleak.csv",
        "metric_start_date": "2023-02-10",
        "history_start_date": "2023-02-10",
        "included_pairs": INCLUDED_PAIRS,
        "initial_pair": "pca_covariance__diversified_risk_parity",
        "initial_hold_days": 0,
        "hold_gate_days": 0,
        "switch_review_interval": 10,
        "cooldown_missing_rank_is_poor": False,
        "fallback_switch_when_all_recent_ranks_poor": False,
        "apply_switch_penalty_to_returns": False,
        "switch_cost_penalty": 0.0,
        "use_switch_hurdle_filter": False,
        "candidate_sort_drawdown_consistency": True,
        "drawdown_consistency_target": 0.07,
        "require_recent_momentum_improving": False,
        "excluded_pairs_containing": ["minimum_variance", "min_variance", "mean_variance", "buy_and_hold"],
        "candidate_top_n": 12,
        "overall_rank_window": 60,
        "overall_rank_top_k": 12,
        "recent_rank_windows": [100, 60, 20, 10],
        "sharpe_rank_tie_band": 0.05,
        "regime_rank_top_n": 12,
    }
    summary = []
    for label, model in MODEL_RUNS:
        out_path = outdir / f"ai_portfolio_router_{label}.parquet"
        print(f"RUN {label} {model}", flush=True)
        try:
            frame = run_ai_portfolio_router(
                manifest,
                out_path,
                params=dict(base_params, model=model),
                window=63,
                regime_rank_window=63,
                start_date="2024-02-09",
            )
            row = metrics(frame)
            row.update(
                {
                    "label": label,
                    "model": model,
                    "path": str(out_path),
                    **audit_no_leak(frame, manifest),
                }
            )
            export_paper_files(label, frame)
            summary.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        except Exception as exc:
            row = {"label": label, "model": model, "path": str(out_path), "error": str(exc)}
            summary.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    out_json = outdir / "ai_portfolio_more_model_sweep_summary.json"
    out_csv = outdir / "ai_portfolio_more_model_sweep_summary.csv"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"SAVED {out_json}", flush=True)


if __name__ == "__main__":
    main()
