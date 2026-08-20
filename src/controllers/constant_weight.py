"""Constant weight controller for baseline."""


class ConstantWeight:
    """Always returns a fixed weight (default 1.0 for buy-and-hold)."""
    
    def __init__(self, params=None):
        params = params or {}
        self.weight = params.get("weight", 1.0)
    
    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        """Return constant weight."""
        return self.weight
