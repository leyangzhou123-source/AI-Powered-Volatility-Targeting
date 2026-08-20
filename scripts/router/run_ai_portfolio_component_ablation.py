"""Run multi-asset AI router component ablations on the full clean pair list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.run_ai_portfolio_random_pool_ablation import (  # noqa: E402
    EXCLUDED_CONTROLLERS,
    base_params,
    cvar95,
    max_drawdown,
    pair_pretest_metrics,
)
from src.router.ai_portfolio_regime_router import run_ai_portfolio_router  # noqa: E402


ABLATIONS = [
    ("no_regime_context", {"disable_regime_context": True}),
    ("no_recent_rank_context", {"disable_recent_rank_context": True}),
    ("no_train_baseline_champion", {"disable_deterministic_baseline_context": True}),
    ("deterministic_switch_decision", {"deterministic_switch_decision": True}),
    ("deterministic_pair_selection", {"deterministic_pair_selection": True}),
]


def eligible_manifest_pool(manifest_path: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    eligible = manifest.loc[
        manifest["status"].eq("ok") & ~manifest["controller"].isin(EXCLUDED_CONTROLLERS)
    ].copy()
    return eligible.reset_index(drop=True)


def pretest_top_pairs(manifest: pd.DataFrame, test_start: str, top_n: int = 30) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for row in manifest.to_dict("records"):
        metrics = pair_pretest_metrics(str(row["path"]), test_start)
        rows.append({"name": row["name"], **metrics})
    ranked = pd.DataFrame(rows).sort_values(
        ["ann_return", "sharpe", "max_dd", "turnover", "name"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    return ranked["name"].head(min(top_n, len(ranked))).astype(str).tolist(), ranked


def summarize(frame: pd.DataFrame, label: str, pool_size: int, path: Path) -> dict:
    ret = frame["returns_with_rf"].astype(float)
    equity = frame["equity_curve_with_rf"].astype(float)
    equity = equity / float(equity.iloc[0])
    active = frame["active_pair"].astype(str)
    ann_return = float(equity.iloc[-1] ** (252.0 / len(frame)) - 1.0)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(252.0))
    return {
        "label": label,
        "pool_size": int(pool_size),
        "path": str(path),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else float("nan"),
        "max_dd": max_drawdown(equity),
        "cvar95": cvar95(ret),
        "total_return": float(equity.iloc[-1] - 1.0),
        "switches": int((active != active.shift(1)).sum() - 1),
        "pairs_used": int(active.nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/multi_asset_tuned_pairs_vol10/manifest.csv")
    parser.add_argument("--outdir", default="results/multi_asset_tuned_pairs_vol10/component_ablation_oss20b_full91")
    parser.add_argument("--test-start", default="2024-02-09")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = eligible_manifest_pool(args.manifest)
    full_clean_pool = manifest["name"].astype(str).tolist()
    included_pairs, pretest_rank = pretest_top_pairs(manifest, args.test_start, top_n=30)
    initial_pair = included_pairs[0]
    (outdir / "full_clean91_pool.json").write_text(
        json.dumps(full_clean_pool, indent=2),
        encoding="utf-8",
    )
    (outdir / "included_pairs_pretest_top30.json").write_text(
        json.dumps(included_pairs, indent=2),
        encoding="utf-8",
    )
    pretest_rank.to_csv(outdir / "pretest_rank_full_clean91.csv", index=False)
    (outdir / "excluded_controllers.json").write_text(
        json.dumps(sorted(EXCLUDED_CONTROLLERS), indent=2),
        encoding="utf-8",
    )
    summaries: list[dict] = []

    for label, overrides in ABLATIONS:
        out_path = outdir / f"ai_portfolio_router_component_ablation_{label}.parquet"
        params = base_params()
        params.update(
            {
                "included_pairs": included_pairs,
                "initial_pair": initial_pair,
                "progress_path": str(outdir / f"progress_{label}.json"),
            }
        )
        params.update(overrides)
        (outdir / f"params_{label}.json").write_text(
            json.dumps(params, indent=2),
            encoding="utf-8",
        )
        if out_path.exists():
            print(f"REUSE {label}: {out_path}", flush=True)
            frame = pd.read_parquet(out_path)
        else:
            print(
                f"RUN {label}: full_pool={len(full_clean_pool)} ai_available_pairs={len(included_pairs)} "
                f"initial_pair={initial_pair}",
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
        summary = summarize(frame, label, len(included_pairs), out_path)
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(outdir / "summary.csv", index=False)
        (outdir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)

    print(f"DONE wrote {len(summaries)} runs to {outdir}", flush=True)


if __name__ == "__main__":
    main()
