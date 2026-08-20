"""Run the multi-asset AI portfolio router with Mixtral."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.run_multi_asset_ai_more_model_sweep import MODEL_RUNS, main


MODEL_RUNS[:] = [("mixtral_8x7b", "mistralai/mixtral-8x7b-instruct-v0.1")]


if __name__ == "__main__":
    main()
