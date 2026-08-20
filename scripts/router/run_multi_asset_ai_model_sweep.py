"""Run the multi-asset AI portfolio router across several model backends."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.router.ai_portfolio_regime_router import load_pair_results, run_ai_portfolio_router


MODEL_RUNS = [
    ("llama_8b", "meta/llama-3.1-8b-instruct"),
    ("qwen_3", "qwen/qwen3-next-80b-a3b-instruct"),
    ("deepseek_r1", "deepseek-ai/deepseek-r1"),
    ("deepseek_v4_pro", "deepseek-ai/deepseek-v4-pro"),
]


INCLUDED_PAIRS = [
    "pca_covariance__diversified_risk_parity",
    "pca_covariance__equal_risk_contribution",
    "pca_covariance__regime_aware_risk_budget",
    "pca_covariance__hysteresis_portfolio",
    "downside_covariance__hysteresis_portfolio",
    "shrunk_sample_covariance__equal_weight",
    "downside_covariance__inverse_vol",
    "pca_covariance__inverse_vol",
    "dynamic_blend_covariance__equal_weight",
    "expanding_covariance__equal_weight",
    "regime_switching_covariance__equal_weight",
    "ewma_covariance__equal_weight",
    "downside_covariance__equal_weight",
    "ledoit_wolf_covariance__hysteresis_portfolio",
    "shrunk_sample_covariance__hysteresis_portfolio",
    "diagonal_ewma_covariance__hysteresis_portfolio",
    "ledoit_wolf_covariance__inverse_vol",
    "ledoit_wolf_covariance__equal_weight",
    "diagonal_ewma_covariance__regime_aware_risk_budget",
    "diagonal_ewma_covariance__inverse_vol",
    "diagonal_ewma_covariance__diversified_risk_parity",
    "diagonal_ewma_covariance__equal_risk_contribution",
    "sample_covariance__equal_weight",
    "robust_median_covariance__equal_weight",
    "shrunk_sample_covariance__inverse_vol",
    "pca_covariance__equal_weight",
    "rolling_corr_ewma_vol__equal_weight",
    "ewma_covariance__hysteresis_portfolio",
    "diagonal_ewma_covariance__equal_weight",
    "expanding_covariance__hysteresis_portfolio",
]


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    r = frame["returns_with_rf"].fillna(0.0)
    eq = np.exp(r.cumsum())
    dd = eq / eq.cummax() - 1.0
    ann_return = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cvar = float(r[r <= r.quantile(0.05)].mean())
    out = {
        "rows": int(len(frame)),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "cvar_95": cvar,
        "total_return": float(eq.iloc[-1] - 1.0),
        "switches": int(frame["switch_executed"].fillna(False).sum()),
        "pairs_used": int(frame["active_pair"].nunique()),
    }
    for col in ("switch_review_error", "decision_error"):
        if col in frame:
            out[f"{col}_count"] = int((frame[col].fillna("").astype(str) != "").sum())
    if "selection_ai_success" in frame:
        out["executed_ai_switches"] = int(
            (frame["switch_executed"].fillna(False) & frame["selection_ai_success"].fillna(False)).sum()
        )
    if "switch_review_ai_success" in frame:
        out["switch_review_ai_success_count"] = int(frame["switch_review_ai_success"].fillna(False).sum())
    return out


def audit_no_leak(frame: pd.DataFrame, manifest: str) -> dict[str, int]:
    frames = load_pair_results(manifest)
    if "date" not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    violations = 0
    if "router_signal_as_of_date" in frame:
        as_of = pd.to_datetime(frame["router_signal_as_of_date"])
        violations += int((as_of >= frame["date"]).sum())
    checked = 0
    for _, row in frame.iterrows():
        pair = str(row["active_pair"])
        date = pd.Timestamp(row["date"])
        if pair not in frames or date not in frames[pair].index:
            violations += 1
            continue
        expected = float(frames[pair].loc[date, "returns_with_rf"])
        actual = float(row.get("raw_returns_with_rf", row["returns_with_rf"]))
        checked += 1
        if abs(expected - actual) > 1e-12:
            violations += 1
    return {"violations": int(violations), "checked_selected_pair_returns": int(checked)}


def export_paper_files(label: str, frame: pd.DataFrame) -> None:
    outdir = Path("paper result folder")
    outdir.mkdir(parents=True, exist_ok=True)
    if "date" not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
    eq_cols = [
        c
        for c in [
            "date",
            "router_signal_as_of_date",
            "active_pair",
            "portfolio_regime",
            "returns_with_rf",
            "equity_curve_with_rf",
            "switch_executed",
            "decision_action",
            "decision_pair",
            "decision_reason",
        ]
        if c in frame.columns
    ]
    sw_cols = [
        c
        for c in [
            "date",
            "router_signal_as_of_date",
            "active_pair",
            "decision_pair",
            "decision_reason",
            "switch_review_action",
            "switch_review_reason",
            "selection_ai_success",
            "switch_review_ai_success",
            "decision_error",
            "switch_review_error",
        ]
        if c in frame.columns
    ]
    frame[eq_cols].to_csv(outdir / f"emm_volrouter_drawdown_consistency_{label}_equity_curve.csv", index=False)
    frame[eq_cols].to_parquet(outdir / f"emm_volrouter_drawdown_consistency_{label}_equity_curve.parquet", index=False)
    frame.loc[frame["switch_executed"].fillna(False), sw_cols].to_csv(
        outdir / f"emm_volrouter_drawdown_consistency_{label}_switch_log.csv",
        index=False,
    )
    frame.loc[frame["switch_executed"].fillna(False), sw_cols].to_parquet(
        outdir / f"emm_volrouter_drawdown_consistency_{label}_switch_log.parquet",
        index=False,
    )


def main() -> None:
    manifest = "results/multi_asset_tuned_pairs_vol10/manifest.csv"
    outdir = Path("results/multi_asset_tuned_pairs_vol10/model_sweep")
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
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
    for label, model in MODEL_RUNS:
        out_path = outdir / f"ai_portfolio_router_{label}.parquet"
        params = dict(base_params, model=model)
        print(f"RUN {label} {model}", flush=True)
        try:
            frame = run_ai_portfolio_router(
                manifest,
                out_path,
                params=params,
                window=63,
                regime_rank_window=63,
                start_date="2024-02-09",
            )
            m = metrics(frame)
            audit = audit_no_leak(frame, manifest)
            m.update(
                {
                    "label": label,
                    "model": model,
                    "path": str(out_path),
                    "no_leak_violations": int(audit["violations"]),
                    "checked_selected_pair_returns": int(audit["checked_selected_pair_returns"]),
                }
            )
            export_paper_files(label, frame)
            summary.append(m)
            print(json.dumps(m, sort_keys=True), flush=True)
        except Exception as exc:
            err = {"label": label, "model": model, "error": str(exc), "path": str(out_path)}
            summary.append(err)
            print(json.dumps(err, sort_keys=True), flush=True)
    summary_path = outdir / "ai_portfolio_model_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(summary).to_csv(outdir / "ai_portfolio_model_sweep_summary.csv", index=False)
    print(f"SAVED {summary_path}", flush=True)


if __name__ == "__main__":
    main()
