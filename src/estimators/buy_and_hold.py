# src/estimators/buy_and_hold.py
import pandas as pd
from src.estimators.base import Estimator

class BuyAndHold(Estimator):
    """
    Engine uses: weight = target_vol / vol_est
    If we output vol_est = target_vol always -> weight == 1 always.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.target_vol = float(self.params.get("target_vol", 0.10))

    def fit(self, returns: pd.Series):
        return self

    def estimate(self, t, returns=None):
        return self.target_vol
