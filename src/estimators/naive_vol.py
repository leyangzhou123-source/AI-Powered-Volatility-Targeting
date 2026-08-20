import numpy as np

class NaiveVolEstimator:
    def __init__(self, params=None):
        params = params or {}
        self.annualize = params.get("annualize", 252)

    def estimate_window(self, w):
        # w 是过去 roll_window 个 returns_clean
        rv = np.mean(np.square(w))
        return np.sqrt(rv * self.annualize)