"""
Normalized Inverse Variance Scaling Controller
================================================

Implements the inverse variance (1/σ²) position sizing rule found in the
volatility targeting literature, with a long-run normalization that anchors
the average realized volatility to the target level.

Derivation
----------
NaiveScaling uses:
    weight = target_vol / σ
    → realized vol ≈ weight × σ = target_vol  ✓  (exact by construction)

A naive 1/σ² rule:
    weight = target_vol / σ²
    → realized vol ≈ weight × σ = target_vol / σ  ✗  (not anchored)

To fix this, introduce a long-run average vol σ_bar as a normalizing constant:
    weight = target_vol × σ_bar / σ²
    → realized vol ≈ weight × σ = target_vol × σ_bar / σ

When σ = σ_bar (normal regime):  realized vol = target_vol             ✓
When σ > σ_bar (high vol):       realized vol < target_vol  (under-exposed, conservative)
When σ < σ_bar (low vol):        realized vol > target_vol  (over-exposed, but capped)

The realized vol is not guaranteed to equal target_vol on every single day —
no vol-targeting strategy achieves that — but the TIME-AVERAGE realized vol
converges to target_vol as long as σ_bar is estimated consistently.

Why 1/σ² instead of 1/σ?
--------------------------
The 1/σ² rule produces a MORE NON-LINEAR response to vol forecasts:
  σ doubles → naive weight halves, variance weight QUARTERS
This means the strategy cuts positions much more aggressively during vol spikes
(e.g. crisis periods like 2008, 2020), at the cost of running closer to
weight_max during calm periods. Whether this improves Sharpe depends on whether
the vol estimator accurately forecasts regime changes.

σ_bar Estimation
-----------------
σ_bar is estimated as a rolling mean of the vol_estimate series over the past
`sigma_bar_window` steps (default 252 — one year). On the first call before
the window is full, we use the expanding mean as a warm-up. This makes σ_bar
data-driven and avoids hard-coding an assumption about the asset's long-run vol.

Usage in YAML
-------------
    controller:
      class: src.controllers.variance_scaling.VarianceScaling
      params:
        sigma_bar_window: 252    # rolling window for σ_bar estimation
        no_trade_band: 0.05      # same as NaiveScaling
"""

import numpy as np
from collections import deque


class VarianceScaling:
    """
    Normalized inverse-variance controller.

        weight = target_vol × σ_bar / σ²

    where σ_bar is a rolling mean of recent vol estimates.
    Achieves target vol on average; cuts more aggressively than NaiveScaling
    in high-vol regimes.

    Parameters
    ----------
    sigma_bar_window : int
        Rolling window (in trading days) used to estimate long-run average vol.
        Default 252 (one year). Longer windows make σ_bar more stable but
        slower to adapt to structural changes in asset vol.
    no_trade_band : float
        Minimum weight change to trigger rebalance. Default 0.05 (same as
        NaiveScaling) — suppresses excessive turnover from small vol moves.
    eps_vol : float
        Variance floor to prevent division by zero. Default 1e-8.
    """

    def __init__(self, params=None):
        params = params or {}
        self.sigma_bar_window = int(params.get("sigma_bar_window", 252))
        self.no_trade_band    = float(params.get("no_trade_band", 0.05))
        self.eps_vol          = float(params.get("eps_vol", 1e-8))

        # Rolling buffer of recent vol estimates — used to compute σ_bar
        self._vol_history = deque(maxlen=self.sigma_bar_window)

    def compute_weight(
        self,
        target_vol: float,
        vol_estimate: float,
        prev_weight: float,
    ) -> float:
        """
        Compute target weight using normalized inverse variance scaling.

        Parameters
        ----------
        target_vol   : annualised target vol (e.g. 0.10)
        vol_estimate : annualised vol forecast from the estimator (σ, not σ²)
        prev_weight  : weight from the previous period

        Returns
        -------
        float : new target weight (before min/max clipping by the engine)
        """
        # Invalid vol estimate → hold previous weight
        if vol_estimate is None or not np.isfinite(vol_estimate) or vol_estimate <= 0:
            return prev_weight

        # Update rolling history for σ_bar
        self._vol_history.append(vol_estimate)

        # σ_bar: rolling mean of recent vol estimates (expanding during warm-up)
        sigma_bar = float(np.mean(self._vol_history))

        # Safety: if σ_bar is somehow invalid, fall back to current estimate
        if not np.isfinite(sigma_bar) or sigma_bar <= 0:
            sigma_bar = vol_estimate

        # σ² with variance floor
        variance_estimate = max(vol_estimate ** 2, self.eps_vol)

        # Core formula: target_vol × σ_bar / σ²
        # Derivation: ensures E[weight × σ] ≈ target_vol when σ ≈ σ_bar
        raw_weight = (target_vol * sigma_bar) / variance_estimate

        # No-trade band: suppress rebalance if change is negligible
        if abs(raw_weight - prev_weight) < self.no_trade_band:
            return prev_weight

        return raw_weight
    def update(self, *args, **kwargs) -> float:
        """
        Compatibility wrapper.

        Supports common calling styles, e.g.
          update(target_vol, vol_estimate, prev_weight)
          update(env, target_vol, vol_estimate, prev_weight)
          update(..., vol_estimate=<>, prev_weight=<>, target_vol=<>)

        Returns the new (raw) weight. Engine will apply min/max clipping.
        """
        # Prefer keyword args if present
        if "target_vol" in kwargs and "vol_estimate" in kwargs:
            target_vol = float(kwargs["target_vol"])
            vol_estimate = float(kwargs["vol_estimate"])
            prev_weight = float(kwargs.get("prev_weight", kwargs.get("weight_prev", 0.0)))
            return self.compute_weight(target_vol, vol_estimate, prev_weight)

        # Otherwise parse positional args by taking the LAST three numeric-ish values
        # as (target_vol, vol_estimate, prev_weight).
        vals = []
        for x in args:
            # skip obvious non-numerics like env objects
            if isinstance(x, (int, float, np.floating)) and np.isfinite(x):
                vals.append(float(x))
        if len(vals) >= 3:
            target_vol, vol_estimate, prev_weight = vals[-3], vals[-2], vals[-1]
            return self.compute_weight(target_vol, vol_estimate, prev_weight)

        # If we can't infer, do a conservative fallback
        return float(kwargs.get("prev_weight", 0.0))