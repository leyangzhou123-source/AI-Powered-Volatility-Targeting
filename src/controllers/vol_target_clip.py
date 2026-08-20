"""Vol targeting with clip bounds and rebalance threshold."""

import numpy as np


class VolTargetClip:
    def __init__(self, params=None):
        p = params or {}
        self.target_vol = float(p.get("target_vol", 0.10))
        self.w_min = float(p.get("w_min", 0.0))
        self.w_max = float(p.get("w_max", 1.5))
        self.rebalance_threshold = float(p.get("rebalance_threshold", 0.05))
        self.eps_vol = float(p.get("eps_vol", 1e-8))

    def reset(self):
        return

    def update(self, **kwargs):
        return

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        tgt = float(target_vol) if target_vol is not None else self.target_vol
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return prev_weight

        raw = tgt / max(float(vol_estimate), self.eps_vol)
        clipped = float(np.clip(raw, self.w_min, self.w_max))
        if abs(clipped - float(prev_weight)) < self.rebalance_threshold:
            return float(prev_weight)
        return clipped
