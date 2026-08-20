import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

# Import all 12 controllers exactly as defined in your src/controllers folder
from src.controllers.constant_weight import ConstantWeight
from src.controllers.cvar_es_targeting import CVaRESTargeting
from src.controllers.drawdown_brake import DrawdownBrake
from src.controllers.drawdown_modulated import DrawdownModulatedController
from src.controllers.hysteresis_controller import HysteresisController
from src.controllers.naive_scaling import NaiveScaling
from src.controllers.priority_stack_controller import PriorityStackController
from src.controllers.regime_controller import RegimeSwitchController
from src.controllers.trend_filter import TrendFilter
from src.controllers.trend_filtered import TrendFilteredController
from src.controllers.variance_scaling import VarianceScaling
from src.controllers.vol_target_clip import VolTargetClip

def calculate_ulcer_index(drawdowns: pd.Series) -> float:
    """Calculates the Ulcer Index (depth and duration of drawdowns)."""
    if len(drawdowns) == 0:
        return 0.0
    return np.sqrt(np.mean(drawdowns**2))

def get_performance_metrics(strategy_returns: pd.Series, strat_name: str) -> dict:
    """Computes institutional performance metrics for a daily return series."""
    rets = strategy_returns.dropna()
    if len(rets) == 0:
        return {"Controller": strat_name}

    ann_factor = 252.0
    ann_return = rets.mean() * ann_factor
    ann_vol = rets.std() * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cum_returns = (1 + rets).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else np.nan
    ulcer_index = calculate_ulcer_index(drawdowns * 100) 

    var_95 = np.percentile(rets, 5)
    cvar_95 = rets[rets <= var_95].mean()
    win_rate = (rets > 0).mean()

    return {
        "Controller": strat_name,
        "Ann. Return": f"{ann_return:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": round(sharpe, 2),
        "Calmar Ratio": round(calmar, 2),
        "Max Drawdown": f"{max_dd:.2%}",
        "Ulcer Index": round(ulcer_index, 2),
        "Hist. VaR (95%)": f"{var_95:.2%}",
        "CVaR/ES (95%)": f"{cvar_95:.2%}",
        "Win Rate": f"{win_rate:.2%}"
    }

def main():
    print("Loading Master Dataset...")
    data_path = Path("data/processed/Master_Dataset.parquet") 
    
    if not data_path.exists():
        print(f"Error: Could not find {data_path}.")
        return

    df = pd.read_parquet(data_path)
    ret_col = 'returns_clean' if 'returns_clean' in df.columns else 'returns'
    
    print("Running Benchmark Estimator (20-Day Realized Volatility)...")
    df['rv20_forecast'] = df[ret_col].rolling(20).std() * np.sqrt(252.0)
    backtest_df = df.dropna(subset=['rv20_forecast']).copy()

    target_vol = 0.15

    # Instantiate controllers using their default param dictionaries
    controllers = {
        "01_ConstantWeight": ConstantWeight(),
        "02_NaiveScaling": NaiveScaling(),
        "03_VarianceScaling": VarianceScaling(),
        "04_VolTargetClip": VolTargetClip(),
        "05_Hysteresis": HysteresisController(),
        "06_TrendFilter": TrendFilter(),
        "07_TrendFiltered": TrendFilteredController(),
        "08_DrawdownBrake": DrawdownBrake(),
        "09_DrawdownModulated": DrawdownModulatedController(),
        "10_CVaRESTargeting": CVaRESTargeting(),
        "11_RegimeSwitch": RegimeSwitchController(),
        "12_PriorityStack": PriorityStackController()
    }

    results = []
    signals_out = {}

    print("\nExecuting Event-Driven Simulation...")
    
    for name, ctrl in controllers.items():
        # Reset controller state for a fresh backtest
        if hasattr(ctrl, 'reset'):
            ctrl.reset()

        weights_history = []
        strat_returns = []
        
        prev_w = 0.0  # Initial weight at t=0
        equity = 1.0  # Cumulative strategy equity

        # Step through time chronologically
        for row in backtest_df.itertuples():
            ret = getattr(row, ret_col)
            vol_est = getattr(row, 'rv20_forecast')

            # 1. Realize return from yesterday's weight decision
            step_ret = prev_w * ret
            strat_returns.append(step_ret)
            
            # 2. Update compounding strategy equity
            equity *= (1.0 + step_ret)

            # 3. Pass today's data to the controller's internal state
            if hasattr(ctrl, 'update'):
                # Using **kwargs handles controllers that only want 'ret' vs 'equity'
                ctrl.update(ret=ret, equity=equity, vol_estimate=vol_est)

            # 4. Compute the target weight for TOMORROW
            if hasattr(ctrl, 'compute_weight'):
                w = ctrl.compute_weight(target_vol, vol_est, prev_w)
            else:
                w = prev_w

            weights_history.append(w)
            prev_w = w

        # Save the chronological weight series
        signals_out[f"{name}_weight"] = weights_history
        
        # Calculate performance metrics for this controller
        metrics = get_performance_metrics(pd.Series(strat_returns, index=backtest_df.index), name)
        results.append(metrics)

    # Output Results
    results_df = pd.DataFrame(results).set_index("Controller")
    
    print("\n--- Controller Performance Tear Sheet ---")
    print(results_df.to_markdown())

    
    # Save to disk
    out_dir = Path("results/controllers")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(out_dir / "controller_metrics_comparison.csv")
    
    # Compile the weights into a DataFrame and align indices
    signals_df = pd.DataFrame(signals_out, index=backtest_df.index)
    
    # We shift the output weights by 1 before saving to explicitly represent 
    # the weight that was APPLIED on a given day, maintaining strict out-of-sample alignment.
    applied_weights_df = signals_df.shift(1).fillna(0.0)
    applied_weights_df.to_parquet(out_dir / "controller_weights.parquet")
    
    print(f"\nSaved metrics and weight signals to {out_dir}/")

if __name__ == "__main__":
    main()