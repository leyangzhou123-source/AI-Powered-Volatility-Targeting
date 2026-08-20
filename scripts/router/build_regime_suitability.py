"""Build regime suitability scores for estimator-controller pairs.

The script reads rolling pair backtests, scores every pair inside each realized
volatility regime, and aggregates those scores to estimator-level and
controller-level suitability tables.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.env import Env


REGIME_ORDER = ["low", "middle", "high"]


def _short_class(path: str) -> str:
    return str(path).rsplit(".", 1)[-1]


def _canonical_regime(value: Any) -> str | None:
    text = str(value).lower()
    if text in {"low", "mid", "middle", "normal", "high"}:
        return "middle" if text in {"mid", "normal"} else text
    return None


def _sharpe(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return 0.0
    vol = float(r.std(ddof=1))
    if vol <= 0 or not np.isfinite(vol):
        return 0.0
    return float((r.mean() / vol) * math.sqrt(252.0))


def _max_drawdown(equity: pd.Series) -> float:
    e = pd.to_numeric(equity, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(e) < 2:
        return 0.0
    dd = e / e.cummax() - 1.0
    return float(-dd.min())


def _normal_rank(series: pd.Series, ascending: bool) -> pd.Series:
    if series.empty:
        return series
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    rank = series.rank(method="average", ascending=ascending)
    return (rank - 1.0) / max(len(series) - 1.0, 1.0)


def _score_regime_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["sharpe_rank"] = _normal_rank(out["sharpe"], ascending=True)
    out["drawdown_rank"] = _normal_rank(out["max_drawdown"], ascending=False)
    out["turnover_rank"] = _normal_rank(out["avg_turnover"], ascending=False)
    out["vol_tracking_rank"] = _normal_rank(out["vol_tracking_error"], ascending=False)
    out["suitability_score"] = (
        0.45 * out["sharpe_rank"]
        + 0.25 * out["drawdown_rank"]
        + 0.15 * out["turnover_rank"]
        + 0.15 * out["vol_tracking_rank"]
    )
    return out


def build_pair_metrics(manifest_path: Path, target_vol: float) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "ok"].copy()

    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        path = Path(str(item["path"]))
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        if "vol_regime" not in df.columns:
            continue

        df = df.copy()
        df["regime"] = df["vol_regime"].map(_canonical_regime)
        df["turnover"] = pd.to_numeric(df["weight"], errors="coerce").diff().abs().fillna(0.0)

        pair = str(item["name"])
        estimator = _short_class(str(item["estimator"]))
        controller = _short_class(str(item["controller"]))

        for regime in REGIME_ORDER:
            sub = df[df["regime"] == regime]
            if sub.empty:
                continue
            realized_vol = float(pd.to_numeric(sub["returns"], errors="coerce").std(ddof=1) * math.sqrt(252.0))
            rows.append(
                {
                    "pair": pair,
                    "estimator": estimator,
                    "controller": controller,
                    "regime": regime,
                    "n_days": int(len(sub)),
                    "sharpe": _sharpe(sub["returns"]),
                    "max_drawdown": _max_drawdown(sub["equity_curve"]),
                    "avg_turnover": float(sub["turnover"].mean()),
                    "realized_vol": realized_vol,
                    "vol_tracking_error": abs(realized_vol - target_vol),
                }
            )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError(f"No usable pair metrics found from {manifest_path}")

    scored = []
    for regime, group in metrics.groupby("regime", sort=False):
        scored.append(_score_regime_rows(group))
    return pd.concat(scored, ignore_index=True)


def aggregate_scores(pair_scores: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        pair_scores.groupby([group_col, "regime"], as_index=False)
        .agg(
            suitability_score=("suitability_score", "mean"),
            sharpe=("sharpe", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            avg_turnover=("avg_turnover", "mean"),
            vol_tracking_error=("vol_tracking_error", "mean"),
            n_pairs=("pair", "nunique"),
        )
        .sort_values(["regime", "suitability_score"], ascending=[True, False])
    )


def build_router_bias(pair_scores: pd.DataFrame, scale: float, top_n: int | None) -> dict[str, dict[str, float]]:
    bias: dict[str, dict[str, float]] = {}
    for regime, group in pair_scores.groupby("regime", sort=False):
        ranked = group.sort_values("suitability_score", ascending=False)
        if top_n is not None and top_n > 0:
            ranked = ranked.head(top_n)
        bias[regime] = {
            str(row["pair"]): round(float(row["suitability_score"]) * scale, 4)
            for _, row in ranked.iterrows()
        }
    return bias


def build_group_bias(
    scores: pd.DataFrame,
    key_col: str,
    scale: float,
    top_n: int | None = None,
) -> dict[str, dict[str, float]]:
    bias: dict[str, dict[str, float]] = {}
    for regime, group in scores.groupby("regime", sort=False):
        ranked = group.sort_values("suitability_score", ascending=False)
        if top_n is not None and top_n > 0:
            ranked = ranked.head(top_n)
        bias[regime] = {
            str(row[key_col]): round(float(row["suitability_score"]) * scale, 4)
            for _, row in ranked.iterrows()
        }
    return bias


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regime suitability scores for router pairs.")
    parser.add_argument("--manifest", default="results/rolling_pairs/manifest.csv")
    parser.add_argument("--output-dir", default="results/evaluation/regime_suitability")
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--bias-scale", type=float, default=0.5)
    parser.add_argument("--top-n-bias", type=int, default=0, help="0 keeps all pairs in YAML bias.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = Env.path("root") / manifest_path

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Env.path("root") / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_scores = build_pair_metrics(manifest_path, target_vol=args.target_vol)
    estimator_scores = aggregate_scores(pair_scores, "estimator")
    controller_scores = aggregate_scores(pair_scores, "controller")
    bias = build_router_bias(
        pair_scores,
        scale=args.bias_scale,
        top_n=args.top_n_bias if args.top_n_bias > 0 else None,
    )

    pair_scores.to_csv(out_dir / "pair_regime_suitability.csv", index=False)
    estimator_scores.to_csv(out_dir / "estimator_regime_suitability.csv", index=False)
    controller_scores.to_csv(out_dir / "controller_regime_suitability.csv", index=False)

    with open(out_dir / "router_regime_bias.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"router": {"params": {"regime_bias": bias}}}, f, sort_keys=False)

    router_suitability = {
        "router": {
            "params": {
                "regime_bias": bias,
                "estimator_regime_bias": build_group_bias(estimator_scores, "estimator", args.bias_scale),
                "controller_regime_bias": build_group_bias(controller_scores, "controller", args.bias_scale),
                "use_heuristic_regime_bias": False,
            }
        }
    }
    with open(out_dir / "router_regime_suitability.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(router_suitability, f, sort_keys=False)

    print(f"Wrote suitability tables to {out_dir}")
    for regime in REGIME_ORDER:
        top = pair_scores[pair_scores["regime"] == regime].nlargest(5, "suitability_score")
        print(f"\nTop pairs for {regime}:")
        print(top[["pair", "suitability_score", "sharpe", "max_drawdown", "avg_turnover"]].to_string(index=False))


if __name__ == "__main__":
    main()
