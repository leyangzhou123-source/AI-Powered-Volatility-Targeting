import pandas as pd
import numpy as np

class MacroQuantileRegime:
    """
    Evaluates current macro indicators against their trailing N-day distribution.
    Outputs a continuous 'Stress Score' between 0.0 and 1.0.
    """
    def __init__(self, lookback_window=756): # 3 years of trading days
        self.lookback_window = lookback_window

    def fit_predict(self, features_df: pd.DataFrame) -> pd.Series:
        """
        features_df: DataFrame containing ['rv20', 'vrp', 't10y2y']
        """
        # We want to measure how extreme today's features are compared to the lookback window.
        # .rank(pct=True) on a rolling window gives us the Empirical CDF (0.0 to 1.0)
        
        # Rank Realized Vol (High is stressful)
        rv_rank = features_df['rv20'].rolling(self.lookback_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Rank VRP (Low/Negative VRP is stressful - means VIX is underpricing risk)
        # We invert it so 1.0 means high stress
        vrp_rank = 1.0 - features_df['vrp'].rolling(self.lookback_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Blend the ranks into a single Continuous Stress Score (Regime Probability)
        # You can weight these however you see fit based on macro intuition
        stress_score = (0.6 * rv_rank) + (0.4 * vrp_rank)
        
        # Optional: Smooth the output to prevent daily jumping
        stress_score = stress_score.rolling(5).mean()
        
        return stress_score