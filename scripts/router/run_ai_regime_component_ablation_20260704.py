"""Run independent AI-regime router component ablations on the full pool."""

from __future__ import annotations

import json
import math
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


OOS_START = "2023-02-10"
OOS_END = "2026-02-10"
MODEL = "openai/gpt-oss-20b"
API_KEY_ENV = "NVAPI_KEY2"
REQUEST_INTERVAL_SECONDS = "4"


RUNS = [
    (
        "no_regime",
        "router_master_ai_regime_oss20b_filtered_strictjson2_no_regime_20230210_20260210",
        "results/evaluation/ai_regime_router_oss20b_filtered_strictjson2_no_regime_20230210_20260210",
    ),
    (
        "no_recent_ranks",
        "router_master_ai_regime_oss20b_filtered_strictjson2_no_recent_ranks_20230210_20260210",
        "results/evaluation/ai_regime_router_oss20b_filtered_strictjson2_no_recent_ranks_20230210_20260210",
    ),
    (
        "no_benchmark_baseline",
        "router_master_ai_regime_oss20b_filtered_strictjson2_no_benchmark_baseline_20230210_20260210",
        "results/evaluation/ai_regime_router_oss20b_filtered_strictjson2_no_benchmark_baseline_20230210_20260210",
    ),
    (
        "deterministic_switch",
        "router_master_ai_regime_oss20b_filtered_strictjson2_deterministic_switch_20230210_20260210",
        "results/evaluation/ai_regime_router_oss20b_filtered_strictjson2_deterministic_switch_20230210_20260210",
    ),
    (
        "deterministic_selection",
        "router_master_ai_regime_oss20b_filtered_strictjson2_deterministic_selection_20230210_20260210",
        "results/evaluation/ai_regime_router_oss20b_filtered_strictjson2_deterministic_selection_20230210_20260210",
    ),
]


def _sortino(returns: pd.Series) -> float:
    returns = returns.astype(float)
    downside = returns[returns < 0]
    if downside.empty:
        return float("nan")
    downside_dev = math.sqrt(float(downside.pow(2).mean())) * math.sqrt(252.0)
    if downside_dev <= 0:
        return float("nan")
    return float(returns.mean() * 252.0 / downside_dev)


def _summarize(output_dir: str) -> dict[str, object]:
    out = Path(output_dir)
    summary = pd.read_csv(out / "summary.csv").iloc[0]
    diag = pd.read_csv(out / "router_oos_diagnostics.csv").iloc[0]
    ts = pd.read_csv(out / "router_vs_bh_timeseries.csv")
    counts = ts["selected_pair"].value_counts()
    return {
        "output_dir": output_dir,
        "sharpe": float(summary["overall_sharpe"]),
        "sortino": _sortino(ts["router_return"]),
        "calmar": float(summary["overall_calmar"]),
        "ann_return": float(summary["overall_ann_return"]),
        "ann_vol": float(summary["overall_ann_vol"]),
        "drawdown": float(summary["overall_drawdown"]),
        "excess_sharpe": float(summary["excess_sharpe"]),
        "ai_calls": int(diag["ai_total_call_count"]),
        "ai_failures": int(diag["ai_failure_count"]),
        "switch_count": int(diag["switch_count"]),
        "unique_selected": int(diag["unique_selected"]),
        "top_selected": str(counts.index[0]) if not counts.empty else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI-regime component ablations.")
    parser.add_argument("--start-at", default="", help="Ablation name to start at.")
    args = parser.parse_args()

    summaries: list[dict[str, object]] = []
    selected_runs = list(RUNS)
    if args.start_at:
        names = [run[0] for run in selected_runs]
        if args.start_at not in names:
            raise ValueError(f"Unknown ablation for --start-at: {args.start_at}")
        selected_runs = selected_runs[names.index(args.start_at) :]

    for ablation, name, output_dir in selected_runs:
        cmd = [
            sys.executable,
            "-u",
            "scripts/router/run_ai_regime_router_all_pairs.py",
            "--name",
            name,
            "--provider",
            "nvidia",
            "--model",
            MODEL,
            "--api-key-env",
            API_KEY_ENV,
            "--request-min-interval-seconds",
            REQUEST_INTERVAL_SECONDS,
            "--router-ablation",
            ablation,
            "--max-ai-regime-calls",
            "0",
            "--max-ai-selection-calls",
            "0",
            "--max-total-ai-calls",
            "0",
            "--ai-regime-interval",
            "10",
            "--ai-selection-interval",
            "10",
            "--candidate-top-n",
            "6",
            "--oos-start-date",
            OOS_START,
            "--oos-end-date",
            OOS_END,
            "--output-dir",
            output_dir,
            "--timeout",
            "90",
        ]
        print(f"[component-ablation] start {ablation} output_dir={output_dir}", flush=True)
        subprocess.run(cmd, check=True)
        summary = _summarize(output_dir)
        summary["ablation"] = ablation
        summaries.append(summary)
        print(
            "[component-ablation] summary "
            + json.dumps(summary, sort_keys=True, default=str),
            flush=True,
        )

    print("[component-ablation] all_summaries")
    print(json.dumps(summaries, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
