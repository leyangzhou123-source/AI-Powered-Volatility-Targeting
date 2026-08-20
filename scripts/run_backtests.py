"""Run volatility targeting strategies."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import yaml

from src.env import Env
from src.backtest import VolTargetEngine


def load_config(config_path):
    """
    Accepts:
    Path object (full path)
    string full path: "configs/strategies/x.yaml"
    filename only: "x.yaml"
    Returns dict."""
    p = Path(config_path)

   
    if not p.exists():
        p = Env.path("strategies") / p.name

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {Path(config_path)}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML loaded as None: {p}")

    return cfg


def run_strategy(config_path):
    """Run a single strategy from config file."""
    cfg = load_config(config_path)
    
    print(f"\n{'='*50}")
    print(f"Running strategy: {cfg['name']}")
    print(f"{'='*50}")
    
    engine = VolTargetEngine.from_config(cfg)
    engine.run()
    engine.save()
    engine.summary()
    
    return engine


def run_all_strategies():
    """Run all strategies."""
    strategies_dir = Env.path("strategies")
    if not strategies_dir.exists():
        print("No strategies found")
        return
    
    configs = list(strategies_dir.glob("*.yaml"))
    print(f"Found {len(configs)} strategies")
    
    engines = {}
    errors = []
    for config_path in configs:
        try:
            engine = run_strategy(config_path)
            if engine is not None:
                engines[config_path.stem] = engine
            else:
                errors.append(config_path.stem)
        except Exception as e:
            print(f"Error running {config_path.name}: {e}")
            errors.append(config_path.stem)
    
    if errors:
        print(f"\n{len(errors)} strategies failed due to missing data: {', '.join(errors)}")
    
    return engines


def main():
    parser = argparse.ArgumentParser(description="Run volatility targeting strategies")
    parser.add_argument(
        "--strategy", "-s",
        help="Path to strategy config file"
    )
    args = parser.parse_args()
    
    if args.strategy:
        run_strategy(args.strategy)
    else:
        run_all_strategies()
    
    print("\n" + "="*50)
    print("Done! Check results/ folder.")
    print("="*50)


if __name__ == "__main__":
    main()
