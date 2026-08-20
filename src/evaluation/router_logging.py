"""Router evaluation logging helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.env import Env


def log_router_evaluation(
    strategy_name: str,
    router_log: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Persist per-step router decisions for post-backtest evaluation."""
    if router_log is None or router_log.empty:
        return Env.path("router_logs") / f"{strategy_name}_router_log.parquet"

    target_dir = output_dir or Env.path("router_logs")
    target_dir.mkdir(parents=True, exist_ok=True)

    out_path = target_dir / f"{strategy_name}_router_log.parquet"
    router_log.to_parquet(out_path, index=False)
    return out_path
