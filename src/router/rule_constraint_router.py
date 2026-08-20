"""Rule + constraints first router (Scheme A).

This router implements the paper's constrained routing variant:

1. Remove pairs that violate hard diagnostics constraints.
2. Rank feasible pairs with a transparent multi-objective score.

It reuses the base router's performance, regime-bias, diagnostic, and switch
penalty components, then adds explicit control-quality penalties for turnover,
volatility tracking error, and estimator loss.
"""

from __future__ import annotations

from typing import Any

from src.router.router import Router
from src.router.strategy_pair import StrategyPair


class RuleConstraintRouter(Router):
    """Filters invalid pairs first, then ranks by weighted multi-objective score."""

    def __init__(self, pairs: list[StrategyPair], params: dict[str, Any] | None = None):
        super().__init__(pairs, params)
        p = self.params

        # Step 1: hard constraints
        self.max_estimator_loss = float(p.get("max_estimator_loss", 1.0))
        self.max_turnover = float(p.get("max_turnover", 1.0))
        self.max_vol_tracking_error = float(p.get("max_vol_tracking_error", 1.0))
        self.max_invalid_rate = float(p.get("max_invalid_rate", 0.25))
        self.max_exception_rate = float(p.get("max_exception_rate", 0.10))
        self.min_feasible_obs = int(p.get("min_feasible_obs", 0))

        # Step 2: weighted score
        self.alpha = float(p.get("alpha", 1.0))
        self.beta = float(p.get("beta", 0.5))
        self.gamma = float(p.get("gamma", 0.5))
        self.eta = float(p.get("eta", 0.5))
        self.kappa = float(p.get("kappa", 1.0))
        self.include_regime_bias = bool(p.get("include_regime_bias", True))

    def _is_feasible(
        self,
        pair: StrategyPair,
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> tuple[bool, str | None]:
        pair_diag = self._pair_diagnostics(pair, diagnostics)
        pair_perf = self._pair_performance_metrics(pair, performance_metrics)

        obs = int(self._safe_float(pair_perf.get("obs", pair_diag.get("obs", 0)), 0.0))
        estimator_loss = self._safe_float(pair_diag.get("estimator_loss", 0.0))
        turnover = self._safe_float(pair_diag.get("turnover", 0.0))
        vol_track_err = self._safe_float(pair_diag.get("vol_tracking_error", 0.0))
        invalid_rate = self._safe_float(pair_diag.get("invalid_rate", 0.0))
        exception_rate = self._safe_float(pair_diag.get("exception_rate", 0.0))

        if obs < self.min_feasible_obs:
            return False, "min_feasible_obs"
        if estimator_loss > self.max_estimator_loss:
            return False, "estimator_loss"
        if turnover > self.max_turnover:
            return False, "turnover"
        if vol_track_err > self.max_vol_tracking_error:
            return False, "vol_tracking_error"
        if invalid_rate > self.max_invalid_rate:
            return False, "invalid_rate"
        if exception_rate > self.max_exception_rate:
            return False, "exception_rate"
        return True, None

    def _score_components(
        self,
        pair: StrategyPair,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> dict[str, float]:
        pair_diag = self._pair_diagnostics(pair, diagnostics)
        pair_perf = self._pair_performance_metrics(pair, performance_metrics)

        obs = int(self._safe_float(pair_perf.get("obs", 0), 0.0))
        if self.use_perf_weight and obs >= self.min_performance_obs:
            sharpe = self._safe_float(pair_perf.get("rolling_sharpe", 0.0))
            drawdown = max(self._safe_float(pair_perf.get("drawdown", 0.0)), 0.0)
            performance = self.alpha * sharpe - self.beta * drawdown
        else:
            performance = 0.0
            drawdown = 0.0

        turnover = self._safe_float(pair_diag.get("turnover", 0.0))
        vol_track_err = self._safe_float(pair_diag.get("vol_tracking_error", 0.0))
        estimator_loss = self._safe_float(pair_diag.get("estimator_loss", 0.0))

        regime_bias = self._regime_bias_score(pair, market_features) if self.include_regime_bias else 0.0
        diagnostic_penalty = self._diagnostic_penalty(pair, diagnostics)
        control_penalty = self.gamma * turnover + self.eta * vol_track_err + self.kappa * estimator_loss
        switch_penalty = self._switch_penalty(pair)

        total = performance + regime_bias - control_penalty - diagnostic_penalty - switch_penalty
        return {
            "performance": float(performance),
            "regime_bias": float(regime_bias),
            "control_penalty": float(control_penalty),
            "diagnostic_penalty": float(diagnostic_penalty),
            "switch_penalty": float(switch_penalty),
            "drawdown": float(drawdown),
            "turnover": float(turnover),
            "vol_tracking_error": float(vol_track_err),
            "estimator_loss": float(estimator_loss),
            "total": float(total),
        }

    def _score_pair(
        self,
        pair: StrategyPair,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> float:
        return self._score_components(pair, market_features, diagnostics, performance_metrics)["total"]

    def select(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        timestamp: Any = None,
    ) -> StrategyPair:
        can_switch = (self._active_since + 1) >= self.sticky_period

        feasible: list[StrategyPair] = []
        blocked: dict[str, str] = {}
        for pair in self.pairs:
            ok, reason = self._is_feasible(pair, diagnostics, performance_metrics)
            if ok:
                feasible.append(pair)
            else:
                blocked[pair.name] = reason or "filtered"

        # If every candidate is blocked, hold the current pair when possible.
        # This keeps the router from making an arbitrary unsafe jump.
        candidates = feasible if feasible else [self._active_pair or self.default_pair]
        best_pair = self._active_pair
        best_score = float("-inf")
        scores: dict[str, float] = {}
        score_components: dict[str, dict[str, float]] = {}

        for pair in candidates:
            components = self._score_components(pair, market_features, diagnostics, performance_metrics)
            s = float(components["total"])
            scores[pair.name] = s
            score_components[pair.name] = components
            if s > best_score:
                best_score = s
                best_pair = pair

        switched = False
        if can_switch and best_pair.name != self._active_pair.name:
            self._active_pair = best_pair
            self._active_since = 0
            switched = True
        else:
            self._active_since += 1

        decision = {
            "timestamp": timestamp,
            "selected_pair": self._active_pair.name,
            "selected_estimator": self._active_pair.estimator_name,
            "selected_controller": self._active_pair.controller_name,
            "switched": switched,
            "scores": scores,
            "score_components": score_components,
            "blocked": blocked,
            "feasible_count": len(feasible),
            "blocked_count": len(blocked),
            "market_features": dict(market_features),
            "performance_metrics": dict(performance_metrics),
        }
        self.decisions.append(decision)
        return self._active_pair
