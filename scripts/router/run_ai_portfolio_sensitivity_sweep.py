"""Run oss-20b multi-asset router across sensitivity levels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.router.ai_portfolio_regime_router import run_ai_portfolio_router


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


SENSITIVITY_LEVELS = ["very_high", "high", "medium", "low", "very_low"]


def base_params() -> dict:
    return {
        "ai_enabled": True,
        "ai_retries": 1,
        "api_key_env": "NVAPI_KEY2",
        "apply_switch_penalty_to_returns": False,
        "candidate_sort_drawdown_consistency": True,
        "candidate_top_n": 12,
        "cooldown_missing_rank_is_poor": False,
        "cooldown_rank_windows": [60, 20, 10],
        "drawdown_consistency_target": 0.06,
        "excluded_pairs_containing": [
            "minimum_variance",
            "min_variance",
            "mean_variance",
            "buy_and_hold",
            "equal_weight",
        ],
        "fallback_switch_when_all_recent_ranks_poor": False,
        "hold_gate_days": 0,
        "history_start_date": "2023-02-10",
        "initial_hold_days": 0,
        "initial_pair": "pca_covariance__diversified_risk_parity",
        "included_pairs": INCLUDED_PAIRS,
        "max_drawdown_target": 0.06,
        "max_output_tokens": 1024,
        "metric_start_date": "2023-02-10",
        "model": "openai/gpt-oss-20b",
        "overall_rank_top_k": 12,
        "overall_rank_window": 60,
        "precomputed_regime_path": (
            "results/multi_asset_tuned_pairs_vol10/"
            "ai_volatility_regime_series_macro_inputs_start20230210_interval10_noleak.csv"
        ),
        "recent_rank_windows": [100, 60, 20, 10],
        "regime_rank_top_n": 12,
        "require_recent_momentum_improving": False,
        "sharpe_rank_tie_band": 0.05,
        "switch_cost_penalty": 0.0,
        "switch_review_interval": 10,
        "timeout": 90,
        "use_switch_hurdle_filter": False,
    }


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def summarize(frame: pd.DataFrame, label: str, path: Path) -> dict:
    ret = frame["returns_with_rf"].astype(float)
    equity = frame["equity_curve_with_rf"].astype(float)
    equity = equity / float(equity.iloc[0])
    active = frame["active_pair"].astype(str)
    cutoff = ret.quantile(0.05)
    ann_return = float(equity.iloc[-1] ** (252.0 / len(frame)) - 1.0)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(252.0))
    return {
        "label": label,
        "path": str(path),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else float("nan"),
        "max_dd": max_drawdown(equity),
        "cvar95": float(ret[ret <= cutoff].mean()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "switches": int((active != active.shift(1)).sum() - 1),
        "pairs_used": int(active.nunique()),
        "sensitivity_hold_days": int(frame.get("sensitivity_hold_active", pd.Series(dtype=bool)).fillna(False).sum())
        if "sensitivity_hold_active" in frame else 0,
    }


def write_switch_log(frame: pd.DataFrame, out_path: Path) -> None:
    active = frame["active_pair"].astype(str)
    changes = frame.loc[active.ne(active.shift(1)), ["active_pair", "portfolio_regime"]].copy()
    changes.insert(0, "date", [str(x)[:10] for x in changes.index])
    changes.to_csv(out_path.with_suffix(".switch_log.csv"), index=False)
    changes.to_parquet(out_path.with_suffix(".switch_log.parquet"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/multi_asset_tuned_pairs_vol10/manifest.csv")
    parser.add_argument("--outdir", default="results/multi_asset_tuned_pairs_vol10/sensitivity_sweep_oss20b_dd6")
    parser.add_argument("--start-date", default="2024-02-09")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for level in SENSITIVITY_LEVELS:
        label = f"sensitivity_{level}"
        out_path = outdir / f"ai_portfolio_router_{label}.parquet"
        params = base_params()
        params["sensitivity"] = level
        params["progress_path"] = str(outdir / f"progress_{label}.json")
        (outdir / f"params_{label}.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        if out_path.exists():
            print(f"REUSE {label}: {out_path}", flush=True)
            frame = pd.read_parquet(out_path)
        else:
            print(f"RUN {label}", flush=True)
            frame = run_ai_portfolio_router(
                args.manifest,
                out_path,
                params=params,
                window=63,
                regime_rank_window=63,
                start_date=args.start_date,
            )
            frame.to_csv(out_path.with_suffix(".csv"), index=True)
        write_switch_log(frame, out_path)
        summary = summarize(frame, label, out_path)
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(outdir / "summary.csv", index=False)
        (outdir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    print(f"DONE wrote {len(summaries)} runs to {outdir}", flush=True)


if __name__ == "__main__":
    main()
