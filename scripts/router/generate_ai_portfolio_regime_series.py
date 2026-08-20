"""Generate reusable AI portfolio-regime series for multi-asset router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.router.ai_portfolio_regime_router import generate_ai_portfolio_regime_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI portfolio regimes at a low-frequency interval.")
    parser.add_argument("--manifest", default="results/multi_asset_tuned_pairs_vol10/manifest.csv")
    parser.add_argument("--out", default="results/multi_asset_tuned_pairs_vol10/ai_portfolio_regime_series_last3y.csv")
    parser.add_argument("--window", type=int, default=63)
    parser.add_argument("--regime-rank-window", type=int, default=63)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--last-years", type=float, default=3.0)
    parser.add_argument("--params-json", default="{}")
    args = parser.parse_args()
    params = json.loads(args.params_json)
    result = generate_ai_portfolio_regime_series(
        args.manifest,
        args.out,
        params=params,
        window=args.window,
        regime_rank_window=args.regime_rank_window,
        interval=args.interval,
        start_date=args.start_date,
        last_years=args.last_years,
    )
    print(f"Saved regime rows={len(result)} calls={int(result['call_made'].sum())} to {args.out}")


if __name__ == "__main__":
    main()
