"""Base class for volatility estimators."""


class Estimator:
    """Base class for volatility estimators."""
    
    def __init__(self, params=None):
        self.params = params or {}
        self._fitted = False
    
    def fit(self, returns):
        """Fit the estimator on historical returns."""
        raise NotImplementedError

    def estimate(self, t, returns):
        """Return estimated volatility (float) at time t."""
        raise NotImplementedError
