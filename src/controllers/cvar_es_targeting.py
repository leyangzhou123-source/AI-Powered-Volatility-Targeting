"""ES/CVaR targeting controller with historical tail estimation."""

from collections import deque

import numpy as np


class CVaRESTargeting:
    def __init__(self, params=None):
        p = params or {}
        self.es_target = float(p.get("es_target", 0.02))
        self.alpha = float(p.get("alpha", 0.95))
        self.lookback = int(p.get("lookback", 252))
        self.w_min = float(p.get("w_min", 0.0))
        self.w_max = float(p.get("w_max", 1.5))
        self.no_trade_band = float(p.get("no_trade_band", 0.05))
        self.eps_es = float(p.get("eps_es", 1e-8))

        self._ret_hist = deque(maxlen=self.lookback)

    def reset(self):
        self._ret_hist.clear()

    def update(self, ret=None, **kwargs):
        if ret is None or not np.isfinite(ret):
            return
        self._ret_hist.append(float(ret))

    def _estimate_es(self):
        if len(self._ret_hist) < max(20, self.lookback // 4):
            return np.nan
        arr = np.array(self._ret_hist, dtype=float)
        losses = -arr
        var_thr = np.quantile(losses, self.alpha)
        tail = losses[losses >= var_thr]
        if tail.size == 0:
            return np.nan
        return float(np.mean(tail))

    def compute_weight(self, target_vol, vol_estimate, prev_weight):
        es_hat = self._estimate_es()
        if not np.isfinite(es_hat) or es_hat <= 0:
            return float(prev_weight)

        raw = self.es_target / max(es_hat, self.eps_es)
        w = float(np.clip(raw, self.w_min, self.w_max))
        if abs(w - float(prev_weight)) < self.no_trade_band:
            return float(prev_weight)
        return w
