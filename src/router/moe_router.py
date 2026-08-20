"""Mixture-of-experts router (Scheme C).

The router computes a soft probability distribution over estimator-controller
pairs, smooths that distribution through time, then executes the highest-weight
pair. It is a soft gating router with a hard trading decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.router.router import Router
from src.router.strategy_pair import StrategyPair


class MixtureOfExpertsRouter(Router):
    """Computes soft pair weights and executes the highest-weight pair each step."""

    def __init__(self, pairs: list[StrategyPair], params: dict[str, Any] | None = None):
        super().__init__(pairs, params)
        p = self.params
        self.temperature = float(p.get("temperature", 0.5))
        self.momentum = float(p.get("momentum", 0.7))
        self.switch_margin = float(p.get("switch_margin", 0.0))
        self.score_clip = float(p.get("score_clip", 10.0))
        self.include_regime_bias = bool(p.get("include_regime_bias", True))

        self.lambda_risk = float(p.get("lambda_risk", self.lambda_drawdown))
        self.lambda_cost = float(p.get("lambda_cost", 0.3))
        self.lambda_vol_tracking = float(p.get("lambda_vol_tracking", 0.5))
        self.lambda_estimator_loss = float(p.get("lambda_estimator_loss", 0.25))
        self.lambda_diag = float(p.get("lambda_diag", 1.0))

        self.max_invalid_rate = float(p.get("max_invalid_rate", 1.0))
        self.max_exception_rate = float(p.get("max_exception_rate", 1.0))

        self._prev_probs = {pair.name: 1.0 / len(self.pairs) for pair in self.pairs}

    def _is_feasible(self, pair: StrategyPair, diagnostics: dict[str, Any]) -> tuple[bool, str | None]:
        pair_diag = self._pair_diagnostics(pair, diagnostics)
        invalid_rate = self._safe_float(pair_diag.get("invalid_rate", 0.0))
        exception_rate = self._safe_float(pair_diag.get("exception_rate", 0.0))
        if invalid_rate > self.max_invalid_rate:
            return False, "invalid_rate"
        if exception_rate > self.max_exception_rate:
            return False, "exception_rate"
        return True, None

    def _raw_score(
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
            performance = sharpe - self.lambda_risk * drawdown
        else:
            performance = 0.0

        global_vol_track_err = self._safe_float(performance_metrics.get("vol_tracking_error", 0.0))
        turnover = self._safe_float(pair_diag.get("turnover", 0.0))
        vol_track_err = self._safe_float(pair_diag.get("vol_tracking_error", global_vol_track_err))
        estimator_loss = self._safe_float(pair_diag.get("estimator_loss", 0.0))
        regime_bias = self._regime_bias_score(pair, market_features) if self.include_regime_bias else 0.0
        diagnostic_penalty = self.lambda_diag * self._diagnostic_penalty(pair, diagnostics)
        switch_penalty = self._switch_penalty(pair)
        control_penalty = (
            self.lambda_cost * turnover
            + self.lambda_vol_tracking * vol_track_err
            + self.lambda_estimator_loss * estimator_loss
        )

        total = performance + regime_bias - control_penalty - diagnostic_penalty - switch_penalty
        total = float(np.clip(total, -self.score_clip, self.score_clip))
        return {
            "performance": float(performance),
            "regime_bias": float(regime_bias),
            "control_penalty": float(control_penalty),
            "diagnostic_penalty": float(diagnostic_penalty),
            "switch_penalty": float(switch_penalty),
            "turnover": float(turnover),
            "vol_tracking_error": float(vol_track_err),
            "estimator_loss": float(estimator_loss),
            "total": total,
        }

    def _softmax(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        arr = np.array(list(scores.values()), dtype=float)
        t = max(self.temperature, 1e-6)
        z = (arr - np.max(arr)) / t
        ez = np.exp(z)
        probs = ez / np.sum(ez)
        return {k: float(v) for k, v in zip(scores.keys(), probs)}

    @staticmethod
    def _entropy(probs: dict[str, float]) -> float:
        if not probs:
            return 0.0
        arr = np.array(list(probs.values()), dtype=float)
        arr = arr[arr > 0]
        return float(-(arr * np.log(arr)).sum())

    def select(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        timestamp: Any = None,
    ) -> StrategyPair:
        raw_scores: dict[str, float] = {}
        score_components: dict[str, dict[str, float]] = {}
        blocked: dict[str, str] = {}

        for pair in self.pairs:
            ok, reason = self._is_feasible(pair, diagnostics)
            if not ok:
                blocked[pair.name] = reason or "filtered"
                continue
            components = self._raw_score(pair, market_features, diagnostics, performance_metrics)
            raw_scores[pair.name] = components["total"]
            score_components[pair.name] = components

        if not raw_scores and self._active_pair is not None:
            components = self._raw_score(self._active_pair, market_features, diagnostics, performance_metrics)
            raw_scores[self._active_pair.name] = components["total"]
            score_components[self._active_pair.name] = components

        soft_probs = self._softmax(raw_scores)

        # EMA smoothing of router probabilities to reduce switch noise.
        probs: dict[str, float] = {}
        for pair in self.pairs:
            prev = float(self._prev_probs.get(pair.name, 0.0))
            curr = float(soft_probs.get(pair.name, 0.0))
            probs[pair.name] = self.momentum * prev + (1.0 - self.momentum) * curr
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        self._prev_probs = probs

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best_name = ranked[0][0]
        best_prob = float(ranked[0][1])
        second_prob = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        margin = best_prob - second_prob

        if self._active_pair is not None and best_name != self._active_pair.name and margin < self.switch_margin:
            best_name = self._active_pair.name
        best_pair = next(pair for pair in self.pairs if pair.name == best_name)

        can_switch = (self._active_since + 1) >= self.sticky_period
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
            "scores": raw_scores,
            "score_components": score_components,
            "probs": probs,
            "entropy": self._entropy(probs),
            "top_probability": best_prob,
            "probability_margin": margin,
            "blocked": blocked,
            "feasible_count": len(raw_scores),
            "blocked_count": len(blocked),
            "market_features": dict(market_features),
            "performance_metrics": dict(performance_metrics),
        }
        self.decisions.append(decision)
        return self._active_pair
