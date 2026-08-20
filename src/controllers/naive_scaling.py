"""Naive scaling controller."""

import numpy as np


class NaiveScaling:

    def __init__(self, params=None):
        self.no_trade_band = 0.05
        self.eps_vol = 1e-8

    def update(self, vol_estimate, **kwargs):
        return
    
    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return prev_weight
        raw_weight = target_vol / max(vol_estimate, self.eps_vol)
        if abs(raw_weight - prev_weight) < self.no_trade_band:
            return prev_weight

        return raw_weight


