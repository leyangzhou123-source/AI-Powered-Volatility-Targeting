"""Drawdown brake controller: reduce exposure as drawdown deepens."""

import numpy as np


class DrawdownBrake:
    def __init__(self, params=None):
        p = params or {}
        self.target_vol = float(p.get("target_vol", 0.10))
        self.k = float(p.get("k", 1.0))
        self.start_brake_dd = float(p.get("start_brake_dd", 0.10))
        self.full_brake_dd = float(p.get("full_brake_dd", 0.30))
        self.min_brake = float(p.get("min_brake", 0.75))
        self.max_brake = float(p.get("max_brake", 1.0))
        self.w_min = float(p.get("w_min", 0.0))
        self.w_max = float(p.get("w_max", 1.5))
        self.no_trade_band = float(p.get("no_trade_band", 0.05))
        self.eps_vol = float(p.get("eps_vol", 1e-8))

        self._peak_equity = None
        self._equity = None

    def reset(self):
        self._peak_equity = None
        self._equity = None

    def update(self, equity=None, **kwargs):
        if equity is None or not np.isfinite(equity) or equity <= 0:
            return
        self._equity = float(equity)
        if self._peak_equity is None:
            self._peak_equity = self._equity
        else:
            self._peak_equity = max(self._peak_equity, self._equity)

    def _drawdown(self):
        if self._equity is None or self._peak_equity is None or self._peak_equity <= 0:
            return 0.0
        dd = 1.0 - (self._equity / self._peak_equity)
        return float(max(0.0, dd))

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        tgt = float(target_vol) if target_vol is not None else self.target_vol
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return prev_weight

        base = tgt / max(float(vol_estimate), self.eps_vol)
        dd = self._drawdown()
        if dd <= self.start_brake_dd:
            brake = self.max_brake
        else:
            dd_span = max(self.full_brake_dd - self.start_brake_dd, 1e-8)
            scaled_dd = min(1.0, max(0.0, (dd - self.start_brake_dd) / dd_span))
            linear_brake = 1.0 - self.k * scaled_dd
            brake = max(self.min_brake, linear_brake)

        brake = float(np.clip(brake, self.min_brake, self.max_brake))

        w = float(np.clip(base * brake, self.w_min, self.w_max))
        if abs(w - float(prev_weight)) < self.no_trade_band:
            return float(prev_weight)
        return w
