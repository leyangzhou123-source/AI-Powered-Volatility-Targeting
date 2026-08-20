"""Crypto-aware volatility targeting controllers."""

from __future__ import annotations

from collections import deque

import numpy as np


class VolatilityShockThrottle:
    """Vol-target controller that cuts exposure faster during volatility jumps."""

    def __init__(self, params=None):
        p = params or {}
        self.no_trade_band = float(p.get("no_trade_band", 0.025))
        self.vol_window = int(p.get("vol_window", 30))
        self.shock_multiplier = float(p.get("shock_multiplier", 1.75))
        self.shock_cut = float(p.get("shock_cut", 0.50))
        self.max_step_up = float(p.get("max_step_up", 0.20))
        self.max_step_down = float(p.get("max_step_down", 0.75))
        self.eps_vol = float(p.get("eps_vol", 1e-8))
        self._vol_history = deque(maxlen=self.vol_window)

    def reset(self):
        self._vol_history.clear()

    def update(self, vol_estimate=None, **kwargs):
        if vol_estimate is not None and np.isfinite(vol_estimate) and vol_estimate > 0:
            self._vol_history.append(float(vol_estimate))

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return float(prev_weight)

        base_weight = float(target_vol) / max(float(vol_estimate), self.eps_vol)
        if len(self._vol_history) >= 5:
            baseline = float(np.median(self._vol_history))
            if baseline > 0 and float(vol_estimate) >= self.shock_multiplier * baseline:
                base_weight *= self.shock_cut

        delta = base_weight - float(prev_weight)
        if abs(delta) < self.no_trade_band:
            return float(prev_weight)
        if delta > 0:
            delta = min(delta, self.max_step_up)
        else:
            delta = max(delta, -self.max_step_down)
        return float(prev_weight) + float(delta)


class PegAwareVolController:
    """Stablecoin-friendly controller with drawdown and peg-deviation brakes."""

    def __init__(self, params=None):
        p = params or {}
        self.no_trade_band = float(p.get("no_trade_band", 0.01))
        self.peg = float(p.get("peg", 1.0))
        self.peg_deviation_soft = float(p.get("peg_deviation_soft", 0.0015))
        self.peg_deviation_hard = float(p.get("peg_deviation_hard", 0.0060))
        self.drawdown_soft = float(p.get("drawdown_soft", 0.02))
        self.drawdown_hard = float(p.get("drawdown_hard", 0.08))
        self.max_step = float(p.get("max_step", 0.25))
        self.eps_vol = float(p.get("eps_vol", 1e-8))
        self._equity_peak = None
        self._last_equity = None
        self._last_ret = 0.0

    def reset(self):
        self._equity_peak = None
        self._last_equity = None
        self._last_ret = 0.0

    def update(self, vol_estimate=None, ret=None, equity=None, **kwargs):
        if ret is not None and np.isfinite(ret):
            self._last_ret = float(ret)
        if equity is not None and np.isfinite(equity):
            equity = float(equity)
            self._last_equity = equity
            self._equity_peak = equity if self._equity_peak is None else max(self._equity_peak, equity)

    def _drawdown_multiplier(self, equity: float | None) -> float:
        if equity is None or self._equity_peak is None or self._equity_peak <= 0:
            return 1.0
        dd = max(0.0, 1.0 - float(equity) / self._equity_peak)
        if dd >= self.drawdown_hard:
            return 0.0
        if dd <= self.drawdown_soft:
            return 1.0
        span = max(self.drawdown_hard - self.drawdown_soft, 1e-12)
        return float(1.0 - (dd - self.drawdown_soft) / span)

    def _peg_multiplier(self) -> float:
        deviation = abs(np.expm1(self._last_ret))
        if deviation >= self.peg_deviation_hard:
            return 0.0
        if deviation <= self.peg_deviation_soft:
            return 1.0
        span = max(self.peg_deviation_hard - self.peg_deviation_soft, 1e-12)
        return float(1.0 - (deviation - self.peg_deviation_soft) / span)

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return float(prev_weight)

        raw = float(target_vol) / max(float(vol_estimate), self.eps_vol)
        target = raw * self._peg_multiplier() * self._drawdown_multiplier(self._last_equity)

        delta = target - float(prev_weight)
        if abs(delta) < self.no_trade_band:
            return float(prev_weight)
        delta = float(np.clip(delta, -self.max_step, self.max_step))
        return float(prev_weight) + delta
