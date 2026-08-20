"""Run the LLM router over the full configured pair universe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.backtest.engine import VolTargetEngine  # noqa: E402


DECISION_LAYER_SYSTEM_PROMPT = """Output JSON immediately. Do not explain. Do not reason step by step.
You are the decision layer for a volatility-targeting router.
You only see past intraday volatility, strategy return/drawdown/volatility, RV22 benchmark performance, active pair state, and switch frequency.
Every review, decide whether to hold the current pair or request a switch.
Emphasize the recent_10d_focus field first, then confirm with longer 63d/126d context.
Penalize frequent switching heavily. Request switch only when benchmark underperformance, drawdown, or persistent volatility-stream changes justify it.
Treat realized volatility above 11% or active-pair drawdown above 8% as risk warnings even if returns are acceptable.
If pair_concentration says one pair dominates recent history and it is not clearly beating benchmark/risk targets, request switch.
Return only:
{"action": "hold", "reason": "<short reason>", "confidence": 0.0}
or:
{"action": "switch", "reason": "<short reason>", "confidence": 0.0}
Keep reason under 24 words.
"""


SELECTION_LAYER_SYSTEM_PROMPT = """Output the final JSON immediately. Do not explain. Do not reason step by step.
Choose exactly one estimator-controller pair from the provided candidate names.
You do not receive fallback scores or regime-bias priors; ignore any missing prior/score fields.
Primary objective: minimize turnover and drawdown. Treat both as first-priority risk costs.
Secondary objective: among candidates with competitive turnover and drawdown, maximize return versus the RV22 naive-scaling benchmark after turnover.
Keep annualized realized volatility close to 10%, preferably inside 9% to 11%, as a risk constraint.
Reject high-return candidates if turnover/drawdown is materially worse than alternatives or volatility is materially above 11%.
If hard_risk_filter was applied, trust that filtered candidate universe and choose the best performance tradeoff, not the familiar or dominant pair.
Avoid selecting the dominant recent pair unless it clearly beats alternatives on drawdown, volatility target, and benchmark-relative return.
Return only this strict JSON object:
{"pair": "<candidate name>", "reason": "<short reason>", "confidence": 0.0}
Keep reason under 24 words.
"""


def _max_drawdown(equity: pd.Series) -> float:
    e = pd.to_numeric(equity, errors="coerce").dropna()
    if len(e) < 2:
        return 0.0
    dd = e / e.cummax() - 1.0
    return float(-dd.min())


def _summary(result: pd.DataFrame, router_log: pd.DataFrame) -> dict:
    returns = pd.to_numeric(result["returns"], errors="coerce").fillna(0.0)
    ann_ret = float(returns.mean() * 252.0)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(252.0))
    selected = result["selected_pair"].dropna()
    turnover = pd.to_numeric(result["weight"], errors="coerce").diff().abs().fillna(0.0)

    out = {
        "rows": int(len(result)),
        "total_return": float(result["equity_curve"].iloc[-1] / result["equity_curve"].iloc[0] - 1.0),
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else 0.0,
        "max_drawdown": _max_drawdown(result["equity_curve"]),
        "avg_turnover": float(turnover.mean()),
        "unique_selected_pairs": int(selected.nunique()),
        "top_selected_pairs": selected.value_counts().head(12).to_dict(),
    }
    if not router_log.empty:
        llm_attempt_count = pd.to_numeric(router_log.get("llm_attempt_count"), errors="coerce")
        llm_attempt_max = llm_attempt_count.max()
        out.update(
            {
                "llm_attempt_count": int(llm_attempt_max) if pd.notna(llm_attempt_max) else 0,
                "llm_used_count": int(pd.to_numeric(router_log.get("llm_used"), errors="coerce").fillna(False).sum()),
                "llm_error_count": int(router_log.get("llm_error", pd.Series(dtype=object)).fillna("").astype(str).ne("").sum()),
                "llm_call_reasons": router_log.get("llm_call_reason", pd.Series(dtype=object)).fillna("").value_counts().head(12).to_dict(),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all-pairs LLM router backtest.")
    parser.add_argument("--strategy", default="configs/strategies/router_master.yaml")
    parser.add_argument("--name", default="router_master_llm_all_pairs_intraday_event_tuned")
    parser.add_argument("--mode", default="out_of_sample", choices=["all", "in_sample", "out_of_sample"])
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--max-calls", type=int, default=250)
    parser.add_argument("--candidate-top-n", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--weight-max", type=float, default=None)
    args = parser.parse_args()

    with open(args.strategy, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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
    router_cfg["type"] = "llm"
    router_cfg["class"] = "src.router.llm_router.LLMRouter"
    params = router_cfg.setdefault("params", {})
    params.update(
        {
            "provider": args.provider,
            "api_format": "chat_completions",
            "model": args.model,
            "api_key": os.getenv("NVAPI_KEY") or os.getenv("OPENAI_API_KEY", ""),
            "response_format": {"type": "json_object"},
            "raw_response_debug_dir": "results/evaluation/llm_raw_responses",
            "retry_on_invalid_json": True,
            "retry_candidate_top_n": 10,
            "reasoning_effort": "low",
            "regime_suitability_path": "results/evaluation/regime_suitability_all_pairs",
            "use_heuristic_regime_bias": False,
            "two_stage_decision": True,
            "review_interval": 10,
            "decision_mode": "llm_two_stage",
            "always_call_first": True,
            "min_decision_gap": 21,
            "rv_zscore_trigger": 1.5,
            "rv_change_trigger": 0.25,
            "rv_percentile_trigger": 0.9,
            "strategy_drawdown_trigger": 0.08,
            "active_pair_drawdown_trigger": 0.06,
            "benchmark_underperformance_trigger": 0.02,
            "performance_event_lookback": "trailing_63d",
            "decision_recent_lookback": "trailing_10d",
            "prompt_history_lookbacks": ["trailing_10d", "trailing_63d", "trailing_126d"],
            "sticky_period": 5,
            "perf_weight": 1.5,
            "lambda_drawdown": 0.5,
            "lambda_switch": 0.15,
            "regime_suitability_scale": 0.75,
            "include_scores": False,
            "include_regime_suitability_prompt": False,
            "llm_fallback_score_prompt_weight": 0.15,
            "llm_regime_suitability_prompt_scale": 0.25,
            "turnover_penalty_prompt_weight": 14.0,
            "llm_drawdown_penalty_prompt_weight": 12.0,
            "llm_vol_band_penalty_prompt_weight": 8.0,
            "llm_return_reward_prompt_weight": 0.75,
            "llm_target_vol": 0.10,
            "llm_vol_min": 0.09,
            "llm_vol_max": 0.11,
            "llm_max_pair_drawdown": 0.08,
            "candidate_hard_risk_filter": True,
            "candidate_risk_filter_min_count": 20,
            "llm_hard_vol_max": 0.12,
            "llm_hard_drawdown_max": 0.10,
            "switch_frequency_penalty_weight": 2.0,
            "pair_concentration_window": 252,
            "pair_concentration_threshold": 0.70,
            "benchmark_required_margin": 0.0,
            "decision_system_prompt": DECISION_LAYER_SYSTEM_PROMPT,
            "selection_system_prompt": SELECTION_LAYER_SYSTEM_PROMPT,
            "system_prompt": SELECTION_LAYER_SYSTEM_PROMPT,
            "candidate_top_n": args.candidate_top_n,
            "candidate_rank_mode": "performance_first",
            "max_consecutive_pair_calls": 1,
            "diversity_score_margin": 0.20,
            "recent_choice_window": 12,
            "max_calls": args.max_calls,
            "max_output_tokens": 1024,
            "temperature": 0.0,
            "timeout": args.timeout,
            "fail_open": True,
        }
    )

    engine = VolTargetEngine.from_config(cfg)
    result = engine.run(mode=args.mode)
    path = engine.save()
    summary = _summary(result, engine._router_log_df)
    summary["result_path"] = str(path)
    summary["router_log_path"] = f"results/evaluation/router_logs/{args.name}_router_log.parquet"
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
