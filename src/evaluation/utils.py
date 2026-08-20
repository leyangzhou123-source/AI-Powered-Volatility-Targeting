"""Evaluation utilities."""

import numpy as np
import pandas as pd


def compute_metrics(result, cfg):
    """Compute evaluation metrics for a backtest result."""
    returns = result.returns
    index_level = result.index_level
    
    valid = returns[returns != 0].dropna()
    ann = cfg.vol_ann
    sqrt_ann = np.sqrt(ann)
    n_years = len(valid) / ann
    
    metrics = {}
    
    # Volatility
    metrics["realized_vol"] = float(valid.std() * sqrt_ann)
    metrics["vol_vs_target"] = abs(metrics["realized_vol"] - cfg.target_vol)
    
    # Returns
    metrics["total_return"] = float(index_level.iloc[-1] / index_level.iloc[0] - 1)
    if n_years > 0:
        metrics["cagr"] = float((index_level.iloc[-1] / index_level.iloc[0]) ** (1/n_years) - 1)
    
    # Risk-adjusted
    if valid.std() > 0:
        metrics["sharpe"] = float(valid.mean() * ann / (valid.std() * sqrt_ann))
    
    # Drawdown
    cummax = index_level.cummax()
    dd = (index_level - cummax) / cummax
    metrics["max_drawdown"] = float(-dd.min())
    
    # Ulcer Index
    metrics["ulcer_index"] = float(np.sqrt(np.mean(np.square(dd))))
    
    # Turnover
    metrics["turnover_mean"] = float(result.turnover.mean())
    
    return metrics


def build_comparison_table(results):
    """Build DataFrame comparing metrics across strategies."""
    return pd.DataFrame({
        name: r["metrics"] for name, r in results.items()
    }).T


def compute_rolling_volatility(returns, window=21, annualize=252):
    """Compute rolling annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(annualize)


def compute_drawdown(index_level):
    """Compute drawdown series from index level."""
    cummax = index_level.cummax()
    return (index_level - cummax) / cummax