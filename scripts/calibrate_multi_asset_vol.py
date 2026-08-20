"""Calibrate saved multi-asset pair outputs closer to a realized volatility target."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _metric_row(name: str, df: pd.DataFrame, path: Path, target_vol: float) -> dict[str, Any]:
    r = df["returns_with_rf"].fillna(0.0)
    eq = df["equity_curve_with_rf"]
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    max_dd = float((eq / eq.cummax() - 1.0).min())
    turnover = float(df.get("turnover", pd.Series(0.0, index=df.index)).mean())
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    return {
        "name": name,
        "total_return": total_return,
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "target_vol_abs_error": abs(ann_vol - target_vol),
        "avg_turnover": turnover,
        "path": str(path),
    }


def _infer_cost_bps(df: pd.DataFrame, default_cost_bps: float) -> float:
    if "turnover_cost" not in df or "turnover" not in df:
        return default_cost_bps
    valid = df["turnover"].astype(float) > 1e-12
    if not valid.any():
        return default_cost_bps
    rate = (df.loc[valid, "turnover_cost"].astype(float) / df.loc[valid, "turnover"].astype(float)).median()
    if not np.isfinite(rate) or rate <= 0:
        return default_cost_bps
    return float(rate * 10000.0)


def calibrate_frame(
    df: pd.DataFrame,
    target_vol: float,
    train_days: int,
    max_gross: float,
    default_cost_bps: float,
    min_multiplier: float,
    max_multiplier: float,
) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    returns = out["returns_with_rf"].fillna(0.0)
    train = returns.iloc[: min(train_days, len(returns))]
    train_vol = float(train.std(ddof=1) * np.sqrt(252)) if len(train) > 1 else 0.0
    multiplier = target_vol / train_vol if train_vol > 0 else 1.0
    multiplier = float(np.clip(multiplier, min_multiplier, max_multiplier))

    weight_cols = [c for c in out.columns if c.startswith("weight_")]
    return_cols = [f"return_{c.removeprefix('weight_')}" for c in weight_cols]
    if not weight_cols or any(c not in out.columns for c in return_cols):
        return out, multiplier

    weights = out[weight_cols].astype(float) * multiplier
    gross = weights.abs().sum(axis=1)
    gross_scale = (max_gross / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    weights = weights.mul(gross_scale, axis=0)
    gross = weights.abs().sum(axis=1)
    cash_weight = 1.0 - gross
    asset_returns = out[return_cols].astype(float)
    asset_returns.columns = weight_cols
    risky_leg = (weights * asset_returns).sum(axis=1)
    rf = out.get("rf_daily_return", pd.Series(0.0, index=out.index)).fillna(0.0).astype(float)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost_bps = _infer_cost_bps(out, default_cost_bps)
    turnover_cost = turnover * (cost_bps / 10000.0)
    rf_leg = cash_weight * rf

    out[weight_cols] = weights
    out["gross_exposure"] = gross
    out["cash_weight"] = cash_weight
    out["turnover"] = turnover
    out["turnover_cost"] = turnover_cost
    out["risky_leg_return"] = risky_leg
    out["rf_leg_return"] = rf_leg
    out["returns_with_rf"] = risky_leg + rf_leg - turnover_cost
    out["returns_no_rf"] = risky_leg - turnover_cost
    out["equity_curve_with_rf"] = 1000.0 * np.exp(out["returns_with_rf"].fillna(0.0).cumsum())
    out["equity_curve_no_rf"] = 1000.0 * np.exp(out["returns_no_rf"].fillna(0.0).cumsum())
    out["vol_calibration_multiplier"] = multiplier
    out["vol_calibration_train_vol"] = train_vol
    out["vol_calibration_max_gross"] = max_gross
    return out, multiplier


def calibrate_folder(
    in_dir: str | Path,
    out_dir: str | Path,
    target_vol: float = 0.10,
    train_days: int = 252,
    max_gross: float = 2.0,
    default_cost_bps: float = 5.0,
    min_multiplier: float = 0.5,
    max_multiplier: float = 2.0,
) -> pd.DataFrame:
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(in_dir / "manifest.csv")
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    for row in manifest.to_dict("records"):
        if row.get("status") != "ok":
            continue
        src = Path(str(row["path"]))
        if not src.exists():
            continue
        df = pd.read_parquet(src)
        calibrated, multiplier = calibrate_frame(
            df,
            target_vol=target_vol,
            train_days=train_days,
            max_gross=max_gross,
            default_cost_bps=default_cost_bps,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
        )
        dst = out_dir / src.name
        calibrated.to_parquet(dst)
        record = dict(row)
        record["path"] = str(dst)
        record["calibration_multiplier"] = multiplier
        record["status"] = "ok"
        records.append(record)
        metrics.append(_metric_row(str(row["name"]), calibrated, dst, target_vol))

    out_manifest = pd.DataFrame(records)
    out_manifest.to_csv(out_dir / "manifest.csv", index=False)
    summary = pd.DataFrame(metrics)
    summary["rank_score"] = (
        summary["target_vol_abs_error"]
        + 0.5 * summary["max_drawdown"].abs()
        + 0.02 * summary["avg_turnover"]
        - 0.05 * summary["sharpe"]
    )
    summary = summary.sort_values(["target_vol_abs_error", "rank_score"])
    summary.to_csv(out_dir / "pair_metrics_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate multi-asset pair results toward target realized volatility.")
    parser.add_argument("--in-dir", default="results/multi_asset_tuned_pairs")
    parser.add_argument("--out-dir", default="results/multi_asset_tuned_pairs_vol10")
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--max-gross", type=float, default=2.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    summary = calibrate_folder(
        args.in_dir,
        args.out_dir,
        target_vol=args.target_vol,
        train_days=args.train_days,
        max_gross=args.max_gross,
        default_cost_bps=args.cost_bps,
    )
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
