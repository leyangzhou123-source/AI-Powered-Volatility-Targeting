"""Run oss-20b AI portfolio router after repeatedly removing random pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.router.ai_portfolio_regime_router import run_ai_portfolio_router


EXCLUDED_CONTROLLERS = {
    "src.multi_asset.controllers.EqualWeightController",
    "src.multi_asset.controllers.BuyAndHoldController",
    "src.multi_asset.controllers.MinimumVarianceController",
    "src.multi_asset.controllers.VolCappedMinimumVarianceController",
    "src.multi_asset.controllers.MeanVarianceController",
}


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def cvar95(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    cutoff = returns.quantile(0.05)
    return float(returns[returns <= cutoff].mean())


def pair_pretest_metrics(path: str, test_start: str) -> dict:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    hist = frame.loc[frame.index < pd.Timestamp(test_start)].copy()
    ret = hist["returns_with_rf"].astype(float).dropna()
    if ret.empty:
        return {
            "ann_return": -np.inf,
            "sharpe": -np.inf,
            "max_dd": -np.inf,
            "turnover": np.inf,
        }
    equity = (1.0 + ret).cumprod()
    ann_return = float(equity.iloc[-1] ** (252.0 / len(ret)) - 1.0)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(252.0))
    turnover = float(hist.get("turnover", pd.Series(dtype=float)).astype(float).mean())
    return {
        "ann_return": ann_return,
        "sharpe": ann_return / ann_vol if ann_vol else -np.inf,
        "max_dd": max_drawdown(equity),
        "turnover": turnover if np.isfinite(turnover) else np.inf,
    }


def eligible_manifest_pool(manifest_path: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    ok = manifest["status"].eq("ok")
    eligible = manifest.loc[ok & ~manifest["controller"].isin(EXCLUDED_CONTROLLERS)].copy()
    return eligible.reset_index(drop=True)


def top_pretest_pairs(
    manifest: pd.DataFrame,
    pool: list[str],
    *,
    test_start: str,
    top_n: int,
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    by_name = manifest.set_index("name")
    for name in pool:
        if name not in by_name.index:
            continue
        metrics = pair_pretest_metrics(str(by_name.loc[name, "path"]), test_start)
        rows.append({"name": name, **metrics})
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        ["ann_return", "sharpe", "max_dd", "turnover", "name"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    return ranked["name"].head(min(top_n, len(ranked))).tolist(), ranked


def summarize(frame: pd.DataFrame, label: str, pool_size: int, removed: list[str], path: Path) -> dict:
    ret_col = "router_return" if "router_return" in frame.columns else "returns_with_rf"
    eq_col = "router_equity" if "router_equity" in frame.columns else "equity_curve_with_rf"
    ret = frame[ret_col].astype(float)
    equity = frame[eq_col].astype(float)
    equity = equity / float(equity.iloc[0])
    ann_return = float(equity.iloc[-1] ** (252.0 / len(frame)) - 1.0)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(252.0))
    active = frame["active_pair"].astype(str)
    return {
        "label": label,
        "pool_size": pool_size,
        "path": str(path),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else float("nan"),
        "max_dd": max_drawdown(equity),
        "cvar95": cvar95(ret),
        "total_return": float(equity.iloc[-1] - 1.0),
        "switches": int((active != active.shift(1)).sum() - 1),
        "pairs_used": int(active.nunique()),
        "removed_pairs": removed,
        "remaining_pairs": sorted(active.unique().tolist()),
    }


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
            "buy_and_hold",
            "__mean_variance",
        ],
        "fallback_switch_when_all_recent_ranks_poor": False,
        "hold_gate_days": 0,
        "history_start_date": "2023-02-10",
        "initial_hold_days": 0,
        "initial_pair": "pca_covariance__diversified_risk_parity",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/multi_asset_tuned_pairs_vol10/manifest.csv")
    parser.add_argument("--outdir", default="results/multi_asset_tuned_pairs_vol10/random_pool_ablation_oss20b")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--remove-step", type=int, default=15)
    parser.add_argument("--min-runnable-pairs", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--test-start", default="2024-02-09")
    parser.add_argument("--include-full-pool", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    base = base_params()
    manifest = eligible_manifest_pool(args.manifest)
    pool = manifest["name"].tolist()
    removed_total: list[str] = []
    summaries: list[dict] = []
    round_id = 0
    (outdir / "initial_effective_pool.json").write_text(json.dumps(pool, indent=2), encoding="utf-8")

    while len(pool) >= max(args.min_runnable_pairs, 15):
        if round_id == 0 and args.include_full_pool:
            removed = []
        else:
            removed = rng.sample(pool, min(args.remove_step, len(pool)))
            pool = [pair for pair in pool if pair not in set(removed)]
            removed_total.extend(removed)
            if len(pool) < max(args.min_runnable_pairs, 15):
                break

        selected_pairs, ranked = top_pretest_pairs(
            manifest,
            pool,
            test_start=args.test_start,
            top_n=args.top_n,
        )
        label = f"round{round_id}_pool{len(pool)}_top{len(selected_pairs)}"
        out_path = outdir / f"ai_portfolio_router_{label}.parquet"
        ranked.to_csv(outdir / f"pretest_rank_{label}.csv", index=False)
        (outdir / f"included_pairs_{label}.json").write_text(
            json.dumps(selected_pairs, indent=2),
            encoding="utf-8",
        )
        params = base_params()
        params["included_pairs"] = selected_pairs
        params["initial_pair"] = selected_pairs[0]
        params["progress_path"] = str(outdir / f"progress_{label}.json")

        if out_path.exists():
            print(f"REUSE {label}: existing {out_path}", flush=True)
            frame = pd.read_parquet(out_path)
        else:
            print(
                f"RUN {label}: base_pool_size={len(pool)} selected={len(selected_pairs)} removed_now={removed}",
                flush=True,
            )
            frame = run_ai_portfolio_router(
                args.manifest,
                out_path,
                params=params,
                window=63,
                regime_rank_window=63,
                start_date=args.test_start,
            )
            frame.to_csv(out_path.with_suffix(".csv"), index=True)
        summary = summarize(frame, label, len(pool), list(removed_total), out_path)
        summary["selected_pairs"] = selected_pairs
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(outdir / "summary.csv", index=False)
        (outdir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in [
            "label", "ann_return", "ann_vol", "sharpe", "max_dd", "cvar95", "switches", "pairs_used"
        ]}, indent=2), flush=True)
        round_id += 1

    print(f"DONE wrote {len(summaries)} runs to {outdir}", flush=True)


if __name__ == "__main__":
    main()
