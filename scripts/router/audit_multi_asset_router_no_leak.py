"""Audit multi-asset router outputs for pair-file timing leakage.

The router should not construct volatility-targeting strategies from scratch.
It should read precomputed estimator-controller pair files, use pair history
through the previous available date, select a pair, and only then apply the
selected pair's current row return.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.router.ai_portfolio_regime_router import load_pair_results


def _load_router_output(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        out = pd.read_parquet(path)
    else:
        out = pd.read_csv(path)
    if "date" not in out.columns:
        out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date")


def _selected_pair_col(df: pd.DataFrame) -> str:
    for col in ("active_pair", "selected_pair", "selected_pair_name"):
        if col in df.columns:
            return col
    raise ValueError("Router output needs active_pair, selected_pair, or selected_pair_name.")


def _return_col(df: pd.DataFrame) -> str:
    for col in ("raw_returns_with_rf", "pair_return_with_rf", "returns_with_rf", "strategy_return"):
        if col in df.columns:
            return col
    raise ValueError("Router output needs returns_with_rf, strategy_return, or pair_return_with_rf.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit router output for no-lookahead pair-file usage.")
    parser.add_argument("--manifest", default="results/multi_asset_tuned_pairs_vol10/manifest.csv")
    parser.add_argument("--router-output", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    frames = load_pair_results(args.manifest)
    out = _load_router_output(Path(args.router_output))
    pair_col = _selected_pair_col(out)
    ret_col = _return_col(out)

    violations: list[dict[str, object]] = []
    if "router_signal_as_of_date" in out.columns:
        as_of = pd.to_datetime(out["router_signal_as_of_date"])
        bad = out[as_of >= out["date"]]
        for _, row in bad.iterrows():
            violations.append(
                {
                    "date": str(row["date"].date()),
                    "type": "router_signal_not_past",
                    "detail": str(row.get("router_signal_as_of_date")),
                }
            )

    checked_returns = 0
    for _, row in out.iterrows():
        pair = str(row[pair_col])
        date = pd.Timestamp(row["date"])
        if pair not in frames:
            violations.append({"date": str(date.date()), "type": "unknown_pair", "detail": pair})
            continue
        frame = frames[pair]
        if date not in frame.index:
            violations.append({"date": str(date.date()), "type": "missing_pair_date", "detail": pair})
            continue
        expected = float(frame.loc[date, "returns_with_rf"])
        actual = float(row[ret_col])
        checked_returns += 1
        if abs(expected - actual) > args.tolerance:
            violations.append(
                {
                    "date": str(date.date()),
                    "type": "selected_return_mismatch",
                    "detail": f"{pair}: output={actual:.12g}, pair_file={expected:.12g}",
                }
            )

    print(f"rows={len(out)}")
    print(f"checked_selected_pair_returns={checked_returns}")
    print(f"violations={len(violations)}")
    if violations:
        print(pd.DataFrame(violations).head(50).to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
