"""Run the AI portfolio-regime router on multi-asset pair outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.router.ai_portfolio_regime_router import run_ai_portfolio_router


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI portfolio-regime router for multi-asset results.")
    parser.add_argument("--manifest", default="results/multi_asset_tuned_pairs_vol10/manifest.csv")
    parser.add_argument("--out", default="results/multi_asset_tuned_pairs_vol10/ai_portfolio_router_live.parquet")
    parser.add_argument("--window", type=int, default=63)
    parser.add_argument("--regime-rank-window", type=int, default=63)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--last-years", type=float, default=None)
    parser.add_argument("--params-json", default="{}", help="Router params JSON, e.g. '{\"ai_enabled\": false}'.")
    args = parser.parse_args()
    params = json.loads(args.params_json)
    result = run_ai_portfolio_router(
        args.manifest,
        args.out,
        params=params,
        window=args.window,
        regime_rank_window=args.regime_rank_window,
        start_date=args.start_date,
        last_years=args.last_years,
    )
    print(f"Saved router output rows={len(result)} to {args.out}")


if __name__ == "__main__":
    main()
