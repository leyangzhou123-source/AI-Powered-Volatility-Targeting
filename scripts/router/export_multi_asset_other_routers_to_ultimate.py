"""Export multi-asset non-AI router results to the ultimate dataset format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ULTIMATE_COLUMNS = [
    "date",
    "strategy_equity",
    "strategy_return",
    "buy_hold_equity",
    "buy_hold_return",
    "regime",
    "regime_confidence",
    "regime_prediction_date",
    "regime_reason",
    "selected_pair_name",
    "predicted_volatility",
    "selected_weight",
    "selected_turnover",
    "pair_weight_from_file",
    "pair_return_with_rf",
    "pair_return_no_rf",
    "underlying_asset_return",
]


def _metrics(returns: pd.Series) -> dict[str, float]:
    r = pd.Series(returns).fillna(0.0).astype(float)
    eq = np.exp(r.cumsum())
    ann_return = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    dd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
    q = float(r.quantile(0.05)) if len(r) else 0.0
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else q
    return {
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else 0.0,
        "max_drawdown": abs(min(dd, 0.0)),
        "cvar_95": abs(cvar),
        "n": float(len(r)),
    }


def _load_regime_metadata(path: Path) -> pd.DataFrame:
    regime = pd.read_csv(path)
    regime["date"] = pd.to_datetime(regime["date"])
    regime = regime.sort_values("date")
    rename = {
        "portfolio_regime": "regime",
        "confidence": "regime_confidence",
        "reason": "regime_reason",
    }
    regime = regime.rename(columns=rename)
    if "regime" not in regime:
        regime["regime"] = "balanced"
    if "regime_confidence" not in regime:
        regime["regime_confidence"] = np.nan
    if "regime_reason" not in regime:
        regime["regime_reason"] = ""
    regime["regime_prediction_date"] = regime["date"]
    return regime[["date", "regime", "regime_confidence", "regime_prediction_date", "regime_reason"]]


def export_router(
    router_name: str,
    source_dir: Path,
    ultimate_dir: Path,
    regime_metadata: pd.DataFrame,
    prefix: str,
) -> dict[str, Any]:
    src = source_dir / router_name / "router_vs_bh_timeseries.csv"
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    merged = df.merge(regime_metadata, on="date", how="left", suffixes=("", "_meta"))
    merged["regime"] = merged["regime_meta"].fillna(merged["regime"])
    merged["regime_confidence"] = merged["regime_confidence"].fillna(np.nan)
    merged["regime_prediction_date"] = merged["regime_prediction_date"].fillna(merged["date"])
    merged["regime_reason"] = merged["regime_reason"].fillna("")

    out = pd.DataFrame(
        {
            "date": merged["date"].dt.strftime("%Y-%m-%d"),
            "strategy_equity": merged["strategy_equity"],
            "strategy_return": merged["strategy_return"],
            "buy_hold_equity": merged["buy_hold_equity"],
            "buy_hold_return": merged["buy_hold_return"],
            "regime": merged["regime"],
            "regime_confidence": merged["regime_confidence"],
            "regime_prediction_date": pd.to_datetime(merged["regime_prediction_date"]).dt.strftime("%Y-%m-%d"),
            "regime_reason": merged["regime_reason"],
            "selected_pair_name": merged["selected_pair"],
            "predicted_volatility": merged["selected_realized_vol"],
            "selected_weight": 1.0,
            "selected_turnover": merged["selected_turnover"],
            "pair_weight_from_file": 1.0,
            "pair_return_with_rf": merged["strategy_return"],
            "pair_return_no_rf": merged["strategy_return"],
            "underlying_asset_return": merged["buy_hold_return"],
        },
        columns=ULTIMATE_COLUMNS,
    )

    slug = router_name.replace("contextual_bandit", "bandit").replace("rule_constraint", "rule_based")
    stem = f"{prefix}_{slug}"
    ultimate_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ultimate_dir / f"{stem}.csv"
    parquet_path = ultimate_dir / f"{stem}.parquet"
    metadata_path = ultimate_dir / f"{stem}_metadata.csv"
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)

    summary = _metrics(out["strategy_return"])
    metadata = {
        "dataset": stem,
        "router": router_name,
        "source_file": str(src),
        "rows": len(out),
        "start_date": out["date"].iloc[0] if len(out) else "",
        "end_date": out["date"].iloc[-1] if len(out) else "",
        "switches": int(df["switched"].sum()) if "switched" in df else 0,
        "pairs_used": int(df["selected_pair"].nunique()) if "selected_pair" in df else 0,
        **summary,
    }
    pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="results/evaluation/multi_asset_other_routers_updated_moe_ultimate_source")
    parser.add_argument("--ultimate-dir", default="ultimate result")
    parser.add_argument("--regime-path", default="results/multi_asset_tuned_pairs_vol10/ai_volatility_regime_series_macro_inputs_start20230210_interval10.csv")
    parser.add_argument("--routers", default="base,rule_constraint,contextual_bandit,moe")
    parser.add_argument("--prefix", default="ultimate_multi_asset")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    ultimate_dir = Path(args.ultimate_dir)
    regime_metadata = _load_regime_metadata(Path(args.regime_path))
    rows = []
    for router_name in [item.strip() for item in args.routers.split(",") if item.strip()]:
        rows.append(export_router(router_name, source_dir, ultimate_dir, regime_metadata, args.prefix))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ultimate_dir / f"{args.prefix}_other_routers_metadata.csv", index=False)
    print(comparison[["router", "ann_return", "ann_vol", "sharpe", "max_drawdown", "cvar_95", "switches", "pairs_used"]].to_string(index=False))


if __name__ == "__main__":
    main()
