"""Hysteresis/deadband controller around vol-target weight."""

import numpy as np


class HysteresisController:
    def __init__(self, params=None):
        p = params or {}
        self.target_vol = float(p.get("target_vol", 0.10))
        self.deadband = float(p.get("deadband", 0.05))
        self.w_min = float(p.get("w_min", 0.0))
        self.w_max = float(p.get("w_max", 1.5))
        self.eps_vol = float(p.get("eps_vol", 1e-8))

    def reset(self):
        return

    def update(self, **kwargs):
        return

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        tgt = float(target_vol) if target_vol is not None else self.target_vol
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return float(prev_weight)

        target_w = tgt / max(float(vol_estimate), self.eps_vol)
        target_w = float(np.clip(target_w, self.w_min, self.w_max))

        if abs(target_w - float(prev_weight)) <= self.deadband:
            return float(prev_weight)
        return target_w
