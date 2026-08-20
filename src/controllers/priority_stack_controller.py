"""Priority stack controller: hard risk rules first, then vol target."""

from collections import deque

import numpy as np


class PriorityStackController:
    def __init__(self, params=None):
        p = params or {}

        # Base vol targeting
        self.target_vol = float(p.get("target_vol", 0.10))
        self.w_min = float(p.get("w_min", 0.0))
        self.w_max = float(p.get("w_max", 1.5))
        self.eps_vol = float(p.get("eps_vol", 1e-8))

        # Trend gate
        self.trend_lookback = int(p.get("trend_lookback", 63))
        self.trend_floor = float(p.get("trend_floor", 0.80))

        # Drawdown brake
        self.dd_k = float(p.get("dd_k", 1.0))
        self.dd_start = float(p.get("dd_start", 0.10))
        self.dd_full = float(p.get("dd_full", 0.30))
        self.dd_floor = float(p.get("dd_floor", 0.80))

        # Tail risk gate
        self.es_alpha = float(p.get("es_alpha", 0.95))
        self.es_lookback = int(p.get("es_lookback", 252))
        self.es_limit = float(p.get("es_limit", 0.03))
        self.es_floor = float(p.get("es_floor", 0.80))
        self.combine_mode = str(p.get("combine_mode", "mean")).lower()

        # Trading frictions
        self.deadband = float(p.get("deadband", 0.05))
        self.max_step = float(p.get("max_step", 0.35))

        self._ret_hist = deque(maxlen=max(self.trend_lookback, self.es_lookback))
        self._equity = None
        self._peak_equity = None

    def reset(self):
        self._ret_hist.clear()
        self._equity = None
        self._peak_equity = None

    def update(self, ret=None, equity=None, **kwargs):
        if ret is not None and np.isfinite(ret):
            self._ret_hist.append(float(ret))

        if equity is not None and np.isfinite(equity) and equity > 0:
            self._equity = float(equity)
            if self._peak_equity is None:
                self._peak_equity = self._equity
            else:
                self._peak_equity = max(self._peak_equity, self._equity)

    def _trend_gate(self):
        if len(self._ret_hist) < max(5, self.trend_lookback // 5):
            return 1.0
        arr = np.array(list(self._ret_hist)[-self.trend_lookback :], dtype=float)
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 2 else 0.0
        z = mu / sd if sd > 0 else 0.0
        return 1.0 if z > 0 else float(self.trend_floor)

    def _drawdown_brake(self):
        if self._equity is None or self._peak_equity is None or self._peak_equity <= 0:
            return 1.0
        dd = max(0.0, 1.0 - self._equity / self._peak_equity)
        if dd <= self.dd_start:
            return 1.0
        dd_span = max(self.dd_full - self.dd_start, 1e-8)
        scaled_dd = min(1.0, max(0.0, (dd - self.dd_start) / dd_span))
        return float(np.clip(1.0 - self.dd_k * scaled_dd, self.dd_floor, 1.0))

    def _tail_gate(self):
        if len(self._ret_hist) < max(20, self.es_lookback // 4):
            return 1.0
        arr = np.array(list(self._ret_hist)[-self.es_lookback :], dtype=float)
        losses = -arr
        var_thr = np.quantile(losses, self.es_alpha)
        tail = losses[losses >= var_thr]
        if tail.size == 0:
            return 1.0
        es_hat = float(np.mean(tail))
        if es_hat <= 0:
            return 1.0
        return float(np.clip(self.es_limit / es_hat, self.es_floor, 1.0))

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        tgt = float(target_vol) if target_vol is not None else self.target_vol
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return float(prev_weight)

        base = tgt / max(float(vol_estimate), self.eps_vol)
        base = float(np.clip(base, self.w_min, self.w_max))

        trend_gate = self._trend_gate()
        dd_gate = self._drawdown_brake()
        tail_gate = self._tail_gate()

        if self.combine_mode == "product":
            overlay = trend_gate * dd_gate * tail_gate
        elif self.combine_mode == "mean":
            overlay = float(np.mean([trend_gate, dd_gate, tail_gate]))
        else:
            overlay = min(trend_gate, dd_gate, tail_gate)

        gated = base * overlay
        gated = float(np.clip(gated, self.w_min, self.w_max))

        # Hysteresis/deadband first
        if abs(gated - float(prev_weight)) <= self.deadband:
            return float(prev_weight)

        # Step limiter to avoid sudden turnover spikes
        delta = gated - float(prev_weight)
        if abs(delta) > self.max_step:
            gated = float(prev_weight + np.sign(delta) * self.max_step)
        return float(np.clip(gated, self.w_min, self.w_max))
