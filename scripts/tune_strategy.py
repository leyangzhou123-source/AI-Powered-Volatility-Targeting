import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1]))

from src.backtest import VolTargetEngine
from src.env import Env


def grid_search_lookback(config_path, lookback_list):

    root_dir = Path(__file__).parents[1]
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)  # 如

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    results = []
    equity_curves = {}

    print(f"\nStarting Grid Search: {base_cfg['name']}")
    print(f"Mode: In-Sample Optimization")
    print("-" * 60)

    for lb in lookback_list:
        test_cfg = base_cfg.copy()
        test_cfg['estimator']['params']['lookback'] = lb

        engine = VolTargetEngine.from_config(test_cfg)
        result_df = engine.run(mode="in_sample")

        rets = result_df['returns'].fillna(0)
        sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

        results.append({'lookback': lb, 'sharpe': sharpe})
        equity_curves[lb] = result_df['equity_curve']

        print(f"Lookback: {lb:3d} | Sharpe: {sharpe:.4f} | Final Equity: {result_df['equity_curve'].iloc[-1]:.2f}")

    best_res = max(results, key=lambda x: x['sharpe'])
    best_lb = best_res['lookback']

    print("-" * 60)
    print(f"Best Found: {best_lb} days | 🚀 Running Out-of-Sample Validation...")
    final_cfg = base_cfg.copy()
    final_cfg['estimator']['params']['lookback'] = best_lb
    final_engine = VolTargetEngine.from_config(final_cfg)
    oos_result = final_engine.run(mode="out_of_sample")

    output_img_path = results_dir / f"optimization_{base_cfg['name']}.png"
    plot_results(results, equity_curves, oos_result, best_lb, output_img_path)

    final_engine.summary()


def plot_results(results, equity_curves, oos_result, best_lb, save_path):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Parameter Sensitivity
    lbs = [r['lookback'] for r in results]
    sharpes = [r['sharpe'] for r in results]
    ax1.plot(lbs, sharpes, marker='o', color='#2c3e50', linewidth=2, label='IS Sharpe Ratio')
    ax1.axvline(x=best_lb, color='#e74c3c', linestyle='--', label=f'Optimal: {best_lb}d')
    ax1.set_title('Step 1: Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Lookback Window (Days)')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.legend()

    # Walk-forward Analysis
    ax2.plot(equity_curves[best_lb], label='In-Sample (Train)', color='#3498db', alpha=0.6)
    ax2.plot(oos_result['equity_curve'], label='Out-of-Sample (Test)', color='#27ae60', linewidth=2.5)
    ax2.set_title('Step 2: Walk-forward Validation (IS vs OOS)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nProfessional report saved to: {save_path}")


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config_file = root_dir / "configs" / "strategies" / "realized_vol_20d.yaml"
    test_range = [10, 22, 44, 66, 125, 252] 

    if config_file.exists():
        grid_search_lookback(str(config_file), test_range)
    else:
        print(f"File not found: {config_file}")