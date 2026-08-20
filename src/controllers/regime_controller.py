import numpy as np
from collections import deque

class RegimeSwitchController:

    def __init__(self, params=None):
        p = params or {}
        self.no_trade_band = float(p.get("no_trade_band", 0.05))
        self.eps_vol = float(p.get("eps_vol", 1e-8))

        self.lookback = int(p.get("lookback", 252))
        self.high_vol_q = float(p.get("high_vol_q", 0.8))  
        self.mult_low = float(p.get("mult_low", 1.0))
        self.mult_high = float(p.get("mult_high", 0.5))

        self.w_min = float(p.get("w_min", 0.0))
        self.w_max = float(p.get("w_max", 2.0))

        self._vol_hist = deque(maxlen=self.lookback)

    def update(self, vol_estimate, **kwargs):
        if vol_estimate is not None and np.isfinite(vol_estimate) and vol_estimate > 0:
            self._vol_hist.append(float(vol_estimate))

    def _regime_multiplier(self, vol_estimate):
        if len(self._vol_hist) < max(20, int(0.2 * self.lookback)):
            return self.mult_low  

        threshold = np.quantile(np.array(self._vol_hist), self.high_vol_q)
        return self.mult_high if vol_estimate >= threshold else self.mult_low

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return prev_weight

        mult = self._regime_multiplier(float(vol_estimate))
        raw = mult * (target_vol / max(float(vol_estimate), self.eps_vol))
        raw = float(np.clip(raw, self.w_min, self.w_max))

        if abs(raw - prev_weight) < self.no_trade_band:
            return prev_weight
        return raw
