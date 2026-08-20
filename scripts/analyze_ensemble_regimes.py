"""Analyze saved regime assignments from ensemble runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_regime_table(run_dir: Path) -> pd.DataFrame:
    pq = run_dir / "regime_assignments.parquet"
    csv = run_dir / "regime_assignments.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv, index_col=0)
    else:
        raise FileNotFoundError(f"No regime assignments found in {run_dir}")

    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def main(run_dir: str):
    p = Path(run_dir)
    df = load_regime_table(p)

    counts = df["regime_label"].value_counts(dropna=False).rename("count")
    freq = (counts / counts.sum()).rename("frequency")

    out = pd.concat([counts, freq], axis=1)
    out.to_csv(p / "regime_distribution.csv")

    print("Regime distribution:")
    print(out)
    print(f"Saved: {p / 'regime_distribution.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ensemble regime assignments")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="results/ensemble/<run_name> folder",
    )
    args = parser.parse_args()

    main(args.run_dir)
