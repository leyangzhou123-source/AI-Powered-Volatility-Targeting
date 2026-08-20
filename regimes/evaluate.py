import pandas as pd
import numpy as np

def evaluate_regime_model(regime_series: pd.Series, returns: pd.Series, model_name: str = "HMM"):
    """
    Evaluates a regime model based on separation and stability.
    regime_series: pd.Series of ints (0, 1, 2) indexed by date.
    returns: pd.Series of asset returns indexed by date.
    """
    df = pd.concat([regime_series.rename('regime'), returns.rename('return')], axis=1).dropna()
    
    results = {}
    results['Model Name'] = model_name
    
    # 1. State Separation (Mean & Volatility)
    for state in sorted(df['regime'].unique()):
        state_returns = df[df['regime'] == state]['return']
        
        ann_return = state_returns.mean() * 252
        ann_vol = state_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        results[f'State {int(state)} Ann. Return'] = f"{ann_return:.2%}"
        results[f'State {int(state)} Ann. Vol'] = f"{ann_vol:.2%}"
        results[f'State {int(state)} Sharpe'] = f"{sharpe:.2f}"
        
    # 2. Transition Stability (Whipsaw)
    # Count how many times the regime changes
    regime_changes = (df['regime'] != df['regime'].shift(1)).sum()
    
    # Average duration = total days / number of regime blocks
    avg_duration = len(df) / regime_changes if regime_changes > 0 else len(df)
    
    results['Total Regime Changes'] = regime_changes
    results['Avg Regime Duration (Days)'] = round(avg_duration, 1)
    
    return results