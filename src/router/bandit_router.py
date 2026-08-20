"""Contextual bandit router (Scheme B) using LinUCB-style selection.

Each estimator-controller pair is an arm. At every step the router observes a
market context, scores each arm with LinUCB, subtracts implementation/risk
penalties, and selects the best feasible arm subject to sticky switching.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.router.router import Router
from src.router.strategy_pair import StrategyPair


class ContextualBanditRouter(Router):
    """Treats each strategy pair as an arm and updates online from realized reward."""

    def __init__(self, pairs: list[StrategyPair], params: dict[str, Any] | None = None):
        super().__init__(pairs, params)
        p = self.params
        self.alpha_ucb = float(p.get("alpha_ucb", 0.5))
        self.lambda_risk = float(p.get("lambda_risk", 0.5))
        self.lambda_cost = float(p.get("lambda_cost", 0.2))
        self.lambda_vol_tracking = float(p.get("lambda_vol_tracking", 0.5))
        self.lambda_estimator_loss = float(p.get("lambda_estimator_loss", 0.25))
        self.lambda_diag = float(p.get("lambda_diag", 1.0))
        self.reward_clip = float(p.get("reward_clip", 5.0))
        self.ridge_lambda = float(p.get("ridge_lambda", 1.0))
        self.min_arm_pulls = int(p.get("min_arm_pulls", 0))
        self.include_regime_bias = bool(p.get("include_regime_bias", True))

        self.max_invalid_rate = float(p.get("max_invalid_rate", 1.0))
        self.max_exception_rate = float(p.get("max_exception_rate", 1.0))

        self._d = 13
        self._A: dict[str, np.ndarray] = {pair.name: self.ridge_lambda * np.eye(self._d) for pair in self.pairs}
        self._b: dict[str, np.ndarray] = {pair.name: np.zeros((self._d, 1)) for pair in self.pairs}
        self._counts: dict[str, int] = {pair.name: 0 for pair in self.pairs}

        self._last_pair_name: str | None = None
        self._last_context: np.ndarray | None = None
        self._last_perf_obs: int | None = None
        self._last_switched = False

    def _scaled_context_value(self, value: Any, scale: float = 1.0) -> float:
        v = self._safe_float(value, 0.0)
        if scale <= 0:
            scale = 1.0
        return float(np.tanh(v / scale))

    def _context(self, market_features: dict[str, Any], performance_metrics: dict[str, Any]) -> np.ndarray:
        vol_regime = str(market_features.get("vol_regime", "mid")).lower()
        vals = [
            self._scaled_context_value(market_features.get("rolling_vol", 0.0), 0.25),
            self._scaled_context_value(market_features.get("rolling_mean", 0.0), 0.50),
            self._scaled_context_value(market_features.get("rolling_skew", 0.0), 2.00),
            self._scaled_context_value(performance_metrics.get("rolling_sharpe", 0.0), 2.00),
            self._scaled_context_value(performance_metrics.get("drawdown", 0.0), 0.25),
            self._scaled_context_value(performance_metrics.get("realized_vol", 0.0), 0.25),
            self._scaled_context_value(performance_metrics.get("vol_tracking_error", 0.0), 0.10),
            self._scaled_context_value(market_features.get("window_obs", 0.0), 252.0),
            1.0 if vol_regime == "low" else 0.0,
            1.0 if vol_regime in ("mid", "middle", "normal") else 0.0,
            1.0 if vol_regime == "high" else 0.0,
            1.0 if self._active_pair is not None else 0.0,
            1.0,
        ]
        x = np.array(vals, dtype=float).reshape(-1, 1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def _reward(self, diagnostics: dict[str, Any], performance_metrics: dict[str, Any]) -> float:
        if self._last_pair_name is None:
            return 0.0

        pseudo_pair = next((p for p in self.pairs if p.name == self._last_pair_name), None)
        if pseudo_pair is None:
            return 0.0

        pair_diag = self._pair_diagnostics(pseudo_pair, diagnostics)
        pair_perf = self._pair_performance_metrics(pseudo_pair, performance_metrics)

        sharpe = self._safe_float(pair_perf.get("rolling_sharpe", 0.0))
        drawdown = max(self._safe_float(pair_perf.get("drawdown", 0.0)), 0.0)
        global_vol_track_err = self._safe_float(performance_metrics.get("vol_tracking_error", 0.0))
        turnover = self._safe_float(pair_diag.get("turnover", 0.0))
        vol_track_err = self._safe_float(pair_diag.get("vol_tracking_error", global_vol_track_err))
        estimator_loss = self._safe_float(pair_diag.get("estimator_loss", 0.0))

        reward = (
            sharpe
            - self.lambda_risk * drawdown
            - self.lambda_cost * turnover
            - self.lambda_vol_tracking * vol_track_err
            - self.lambda_estimator_loss * estimator_loss
        )
        if self._last_switched:
            reward -= self.lambda_switch

        return float(np.clip(reward, -self.reward_clip, self.reward_clip))

    def _update_last_arm(self, diagnostics: dict[str, Any], performance_metrics: dict[str, Any]) -> None:
        if self._last_pair_name is None or self._last_context is None:
            return

        obs = int(performance_metrics.get("obs", 0))
        if self._last_perf_obs is not None and obs <= self._last_perf_obs:
            return

        r = self._reward(diagnostics, performance_metrics)
        x = self._last_context
        self._A[self._last_pair_name] += x @ x.T
        self._b[self._last_pair_name] += r * x
        self._counts[self._last_pair_name] += 1

    def _is_feasible(self, pair: StrategyPair, diagnostics: dict[str, Any]) -> tuple[bool, str | None]:
        pair_diag = self._pair_diagnostics(pair, diagnostics)
        invalid_rate = self._safe_float(pair_diag.get("invalid_rate", 0.0))
        exception_rate = self._safe_float(pair_diag.get("exception_rate", 0.0))
        if invalid_rate > self.max_invalid_rate:
            return False, "invalid_rate"
        if exception_rate > self.max_exception_rate:
            return False, "exception_rate"
        return True, None

    def _score_components(
        self,
        pair: StrategyPair,
        x: np.ndarray,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> dict[str, float]:
        A = self._A[pair.name]
        b = self._b[pair.name]
        theta = np.linalg.solve(A, b)
        A_inv_x = np.linalg.solve(A, x)

        expected_reward = float((theta.T @ x)[0, 0])
        uncertainty = float(np.sqrt(max((x.T @ A_inv_x)[0, 0], 0.0)))
        exploration = self.alpha_ucb * uncertainty

        regime_bias = self._regime_bias_score(pair, market_features) if self.include_regime_bias else 0.0
        diagnostic_penalty = self.lambda_diag * self._diagnostic_penalty(pair, diagnostics)
        switch_penalty = self._switch_penalty(pair)
        forced_exploration_bonus = 0.0
        if self._counts[pair.name] < self.min_arm_pulls:
            forced_exploration_bonus = self.alpha_ucb

        total = expected_reward + exploration + regime_bias + forced_exploration_bonus - diagnostic_penalty - switch_penalty
        return {
            "expected_reward": float(expected_reward),
            "exploration": float(exploration),
            "regime_bias": float(regime_bias),
            "forced_exploration_bonus": float(forced_exploration_bonus),
            "diagnostic_penalty": float(diagnostic_penalty),
            "switch_penalty": float(switch_penalty),
            "pull_count": float(self._counts[pair.name]),
            "total": float(total),
        }

    def select(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        timestamp: Any = None,
    ) -> StrategyPair:
        # First update using realized performance since the previous decision.
        self._update_last_arm(diagnostics, performance_metrics)

        x = self._context(market_features, performance_metrics)
        ucb_scores: dict[str, float] = {}
        score_components: dict[str, dict[str, float]] = {}
        blocked: dict[str, str] = {}

        best_pair = self._active_pair
        best_score = float("-inf")

        for pair in self.pairs:
            ok, reason = self._is_feasible(pair, diagnostics)
            if not ok:
                blocked[pair.name] = reason or "filtered"
                continue

            components = self._score_components(pair, x, market_features, diagnostics)
            score = components["total"]
            ucb_scores[pair.name] = score
            score_components[pair.name] = components

            if score > best_score:
                best_score = score
                best_pair = pair

        if not ucb_scores and self._active_pair is not None:
            best_pair = self._active_pair
            components = self._score_components(best_pair, x, market_features, diagnostics)
            ucb_scores[best_pair.name] = components["total"]
            score_components[best_pair.name] = components

        can_switch = (self._active_since + 1) >= self.sticky_period
        switched = False
        if can_switch and best_pair.name != self._active_pair.name:
            self._active_pair = best_pair
            self._active_since = 0
            switched = True
        else:
            self._active_since += 1

        self._last_pair_name = self._active_pair.name
        self._last_context = x
        self._last_perf_obs = int(performance_metrics.get("obs", 0))
        self._last_switched = switched

        decision = {
            "timestamp": timestamp,
            "selected_pair": self._active_pair.name,
            "selected_estimator": self._active_pair.estimator_name,
            "selected_controller": self._active_pair.controller_name,
            "switched": switched,
            "scores": ucb_scores,
            "score_components": score_components,
            "blocked": blocked,
            "feasible_count": len(ucb_scores),
            "blocked_count": len(blocked),
            "market_features": dict(market_features),
            "performance_metrics": dict(performance_metrics),
        }
        self.decisions.append(decision)
        return self._active_pair
