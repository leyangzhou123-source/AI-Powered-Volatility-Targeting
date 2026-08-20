"""Drawdown-Modulated Controller (CPPI-Lite)."""

import numpy as np

class DrawdownModulatedController:
    """
    Applies standard naive volatility scaling, but aggressively deleverages
    the portfolio if the realized equity curve enters a severe drawdown.
    """
    def __init__(self, params=None):
        p = params or {}
        
        # Core params
        self.no_trade_band = float(p.get("no_trade_band", 0.05))
        self.eps_vol = float(p.get("eps_vol", 1e-8))
        self.w_min = float(p.get("w_min", 0.0))
        self.w_max = float(p.get("w_max", 1.5))
        self.min_weight_frac = float(p.get("min_weight_frac", 0.65))

        # Drawdown limits
        self.start_cut_dd = float(p.get("start_cut_dd", 0.10))
        self.max_dd = float(p.get("max_dd", 0.30))
        
        # State
        self.peak_equity = 1.0
        self.current_equity = 1.0

    def update(self, vol_estimate, **kwargs):
        """Capture the daily portfolio equity from the engine to track drawdowns."""
        equity = kwargs.get("equity")
        if equity is not None and np.isfinite(equity) and equity > 0:
            self.current_equity = float(equity)
            self.peak_equity = max(self.peak_equity, self.current_equity)

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        # 1. Handle invalid vol
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return prev_weight

        # 2. Base Naive Weight
        raw_weight = target_vol / max(float(vol_estimate), self.eps_vol)

        # 3. Calculate Current Drawdown
        current_dd = (self.peak_equity - self.current_equity) / self.peak_equity

        # 4. Apply Drawdown Modulator
        if current_dd >= self.max_dd:
            raw_weight *= self.min_weight_frac
        elif current_dd > self.start_cut_dd:
            dd_span = max(self.max_dd - self.start_cut_dd, self.eps_vol)
            progress = (current_dd - self.start_cut_dd) / dd_span
            penalty_factor = 1.0 - progress * (1.0 - self.min_weight_frac)
            raw_weight *= penalty_factor

        # 5. Clip to limits
        raw_weight = float(np.clip(raw_weight, self.w_min, self.w_max))

        # 6. Apply No-Trade Band to suppress micro-trades
        if abs(raw_weight - prev_weight) < self.no_trade_band:
            return prev_weight

        return raw_weight
