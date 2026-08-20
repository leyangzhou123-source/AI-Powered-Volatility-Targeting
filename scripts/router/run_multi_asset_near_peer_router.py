"""Run a no-leak near-peer router over trained high-quality pair alternatives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.router.ai_portfolio_regime_router import load_pair_results


def _metrics(returns: pd.Series) -> dict[str, float]:
    r = pd.Series(returns).fillna(0.0).astype(float)
    if r.empty:
        return {
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "cvar_95": 0.0,
            "total_return": 0.0,
        }
    eq = np.exp(r.cumsum())
    dd = eq / eq.cummax() - 1.0
    ann_return = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    q = float(r.quantile(0.05))
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else 0.0,
        "max_drawdown": float(dd.min()),
        "cvar_95": float(r[r <= q].mean()),
        "total_return": float(eq.iloc[-1] - 1.0),
    }


def _score_from_metrics(
    metrics: dict[str, float],
    return_weight: float,
    drawdown_weight: float,
    vol_te_weight: float,
) -> float:
    return (
        float(metrics["sharpe"])
        + float(return_weight) * float(metrics["ann_return"])
        - float(drawdown_weight) * abs(float(metrics["max_drawdown"]))
        - float(vol_te_weight) * abs(float(metrics["ann_vol"]) - 0.10)
    )


def _select_train_candidates(
    frames: dict[str, pd.DataFrame],
    train_start: str,
    train_end: str,
    top_n: int,
) -> list[str]:
    rows = []
    for name, frame in frames.items():
        hist = frame.loc[train_start:train_end]
        if len(hist) < 100:
            continue
        m = _metrics(hist["returns_with_rf"])
        rows.append({"pair": name, **m})
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        raise ValueError("No train candidates have enough observations.")
    ranked["abs_drawdown"] = ranked["max_drawdown"].abs()
    ranked = ranked.sort_values(["sharpe", "abs_drawdown"], ascending=[False, True])
    return ranked.head(int(top_n))["pair"].tolist()


def run_near_peer_router(
    manifest: str,
    out: str,
    train_start: str,
    train_end: str,
    start_date: str,
    end_date: str,
    top_n: int,
    interval: int,
    lookback: int,
    return_weight: float,
    drawdown_weight: float,
    vol_te_weight: float,
    exclude_contains: list[str],
) -> pd.DataFrame:
    frames = load_pair_results(manifest)
    frames = {
        name: frame
        for name, frame in frames.items()
        if not any(token in name for token in exclude_contains)
    }
    candidates = _select_train_candidates(frames, train_start, train_end, top_n)
    frames = {name: frames[name] for name in candidates}
    common_index = pd.DatetimeIndex(sorted(set.intersection(*[set(df.index) for df in frames.values()])))
    route_index = common_index[
        (common_index >= pd.Timestamp(start_date))
        & (common_index <= pd.Timestamp(end_date))
    ]
    if len(route_index) == 0:
        raise ValueError("No route dates remain.")

    active_pair = candidates[0]
    router_returns: list[float] = []
    records = []
    score_snapshot: dict[str, float] = {}
    active_start_step = 0

    for step, date in enumerate(route_index):
        prior_pair = active_pair
        review_due = step > 0 and step % int(interval) == 0
        reason = "Interval hold"
        if review_due:
            scored = {}
            for pair, frame in frames.items():
                hist = frame.loc[frame.index < date].tail(int(lookback))
                if len(hist) < 20:
                    scored[pair] = -999.0
                else:
                    scored[pair] = _score_from_metrics(
                        _metrics(hist["returns_with_rf"]),
                        return_weight=return_weight,
                        drawdown_weight=drawdown_weight,
                        vol_te_weight=vol_te_weight,
                    )
            score_snapshot = scored
            active_pair = max(scored, key=scored.get)
            reason = f"trained_near_peer_score interval={interval} lookback={lookback}"

        switched = active_pair != prior_pair
        if switched:
            active_start_step = step
        raw_return = float(frames[active_pair].loc[date, "returns_with_rf"])
        router_returns.append(raw_return)
        records.append(
            {
                "date": date,
                "router_signal_as_of_date": str(route_index[step - 1])[:10] if step > 0 else None,
                "active_pair": active_pair,
                "portfolio_regime": "near_peer_rotation",
                "regime_confidence": 1.0,
                "regime_reason": "trained top-N near-peer rotation",
                "decision_action": "switch" if switched else "hold",
                "decision_pair": active_pair,
                "decision_reason": reason,
                "switch_executed": switched,
                "switch_penalty_applied": 0.0,
                "switch_cost_tracked": 0.0,
                "switch_penalty_return_applied": False,
                "switch_review_action": "switch" if switched else "hold",
                "switch_review_reason": reason,
                "regime_changed": False,
                "previous_portfolio_regime": None,
                "cooldown_active": False,
                "active_hold_days": step - active_start_step + 1,
                "cooldown_ranks": "{}",
                "cooldown_poor_flags": "{}",
                "cooldown_all_poor": None,
                "initial_hold_active": False,
                "active_regime_rank": None,
                "active_overall_recent_rank": None,
                "raw_returns_with_rf": raw_return,
                "router_vol_overlay_leverage": 1.0,
                "returns_with_rf": raw_return,
                "equity_curve_with_rf": float(1000.0 * np.exp(np.sum(router_returns))),
                "score_snapshot": repr(score_snapshot),
            }
        )

    result = pd.DataFrame(records).set_index("date")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path)
    result.reset_index().to_csv(out_path.with_suffix(".csv"), index=False)
    m = _metrics(result["returns_with_rf"])
    summary = pd.DataFrame(
        [
            {
                **m,
                "switches": int(result["switch_executed"].sum()),
                "pairs_used": int(result["active_pair"].nunique()),
                "start": route_index.min().date(),
                "end": route_index.max().date(),
                "train_start": train_start,
                "train_end": train_end,
                "candidates": ",".join(candidates),
                "review_interval": int(interval),
                "lookback_points": int(lookback),
                "return_weight": float(return_weight),
                "drawdown_weight": float(drawdown_weight),
                "vol_te_weight": float(vol_te_weight),
            }
        ]
    )
    summary.to_csv(out_path.with_name(out_path.stem + "_summary.csv"), index=False)
    result["active_pair"].value_counts().rename_axis("pair").reset_index(name="days").to_csv(
        out_path.with_name(out_path.stem + "_pair_usage.csv"),
        index=False,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/multi_asset_tuned_pairs_vol10/manifest.csv")
    parser.add_argument("--out", default="results/multi_asset_tuned_pairs_vol10/router_near_peer_train_top3_interval25_lookback40.parquet")
    parser.add_argument("--train-start", default="2023-02-10")
    parser.add_argument("--train-end", default="2024-02-08")
    parser.add_argument("--start-date", default="2024-02-09")
    parser.add_argument("--end-date", default="2026-02-10")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--interval", type=int, default=25)
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--return-weight", type=float, default=0.3)
    parser.add_argument("--drawdown-weight", type=float, default=0.4)
    parser.add_argument("--vol-te-weight", type=float, default=0.0)
    parser.add_argument("--exclude-contains", default="mean_variance,minimum_variance,min_variance,buy_and_hold")
    args = parser.parse_args()
    exclude = [x.strip() for x in args.exclude_contains.split(",") if x.strip()]
    result = run_near_peer_router(
        args.manifest,
        args.out,
        args.train_start,
        args.train_end,
        args.start_date,
        args.end_date,
        args.top_n,
        args.interval,
        args.lookback,
        args.return_weight,
        args.drawdown_weight,
        args.vol_te_weight,
        exclude,
    )
    print(f"Saved near-peer router rows={len(result)} to {args.out}")


if __name__ == "__main__":
    main()
