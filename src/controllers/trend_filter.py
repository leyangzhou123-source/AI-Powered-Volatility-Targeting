"""Trend-gated vol targeting controller."""

from collections import deque

import numpy as np


class TrendFilter:
    def __init__(self, params=None):
        p = params or {}
        self.target_vol = float(p.get("target_vol", 0.10))
        self.lookback = int(p.get("lookback", 63))
        self.gate_mode = str(p.get("gate_mode", "linear")).lower()  # hard | linear
        self.gate_floor = float(p.get("gate_floor", 0.85))
        self.slope = float(p.get("slope", 0.75))
        self.w_min = float(p.get("w_min", 0.0))
        self.w_max = float(p.get("w_max", 1.5))
        self.no_trade_band = float(p.get("no_trade_band", 0.05))
        self.eps_vol = float(p.get("eps_vol", 1e-8))

        self._ret_hist = deque(maxlen=self.lookback)

    def reset(self):
        self._ret_hist.clear()

    def update(self, ret=None, **kwargs):
        if ret is None or not np.isfinite(ret):
            return
        self._ret_hist.append(float(ret))

    def _trend_signal(self):
        if len(self._ret_hist) < max(5, self.lookback // 5):
            return 0.0
        mu = float(np.mean(self._ret_hist))
        sd = float(np.std(self._ret_hist, ddof=1)) if len(self._ret_hist) > 2 else 0.0
        if sd <= 0:
            return 0.0
        return mu / sd

    def _gate(self):
        z = self._trend_signal()
        if self.gate_mode == "linear":
            g = np.clip(0.5 + 0.5 * np.tanh(self.slope * z), self.gate_floor, 1.0)
            return float(g)
        return 1.0 if z > 0 else float(self.gate_floor)

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        tgt = float(target_vol) if target_vol is not None else self.target_vol
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return prev_weight

        base = tgt / max(float(vol_estimate), self.eps_vol)
        gated = base * self._gate()
        w = float(np.clip(gated, self.w_min, self.w_max))
        if abs(w - float(prev_weight)) < self.no_trade_band:
            return float(prev_weight)
        return w
