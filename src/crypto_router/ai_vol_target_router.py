"""Crypto AI regime router for 365-day volatility-targeted pair switching."""

from __future__ import annotations

import json
from typing import Any

from src.router.ai_regime_router import AIRegimeRouter


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _regime_prompt(asset_label: str, target_vol: float) -> str:
    return f"""Output JSON immediately. Do not explain. Do not reason step by step.
You generate regime information for a {asset_label} volatility-targeting router.
Classify the current {asset_label} market into exactly one volatility regime: low, middle, or high.
Use only supplied past information. Prefer persistent realized volatility, intraday realized volatility,
vol-target error versus a {_pct(target_vol)} annualized target, drawdown, benchmark context, and recent regime history.
Return only:
{{"vol_regime": "low", "confidence": 0.0, "reason": "<short reason>"}}
or:
{{"vol_regime": "middle", "confidence": 0.0, "reason": "<short reason>"}}
or:
{{"vol_regime": "high", "confidence": 0.0, "reason": "<short reason>"}}
Keep reason under 24 words.
"""


def _switch_prompt(asset_label: str, target_vol: float, ann_factor: float) -> str:
    return f"""Return JSON only. No reasoning. No prose.
Decide hold or switch for a {asset_label} combined volatility strategy.
The strategy targets {_pct(target_vol)} annualized volatility using a {ann_factor:.0f}-day annualization factor.
Switch when the active pair is materially below/above target volatility, has worse recent Sharpe than challengers,
or the {asset_label} regime/context changed. Hold only when active tracks target volatility and remains competitive.
Return exactly one of:
{{"action":"hold"}}
or
{{"action":"switch"}}
"""


def _selection_prompt(asset_label: str, target_vol: float) -> str:
    return f"""Return JSON only. No reasoning. No prose.
Choose exactly one supplied {asset_label} pair name. The active pair is not supplied and cannot be chosen.
Primary objective: improve risk-adjusted {asset_label} returns while keeping realized volatility close to the {_pct(target_vol)} annual target.
Prefer candidates with lower vol_tracking_error, positive Sharpe edge versus the 20d realized-vol naive baseline,
and controlled drawdown. Turnover is only a tie-breaker.
Return exactly:
{{"pair":"<supplied pair name>"}}
"""


class CryptoVolTargetAIRouter(AIRegimeRouter):
    """Crypto router that keeps AI regime/switch/selection calls and prioritizes vol targeting."""

    def __init__(self, pairs, params: dict[str, Any] | None = None):
        params = dict(params or {})
        params.setdefault("target_vol", 0.35)
        params.setdefault("ann_factor", 365.0)
        target_vol = float(params.get("target_vol", 0.35))
        ann_factor = float(params.get("ann_factor", 365.0))
        asset_label = str(params.get("asset_label", "crypto")).upper()
        params.setdefault("asset_label", asset_label)
        params.setdefault("ai_regime_system_prompt", _regime_prompt(asset_label, target_vol))
        params.setdefault("ai_switch_system_prompt", _switch_prompt(asset_label, target_vol, ann_factor))
        params.setdefault("ai_selection_system_prompt", _selection_prompt(asset_label, target_vol))
        params.setdefault("sensitiveness", "high")
        params.setdefault("sticky_period", 20)
        params.setdefault("ai_selection_interval", 20)
        params.setdefault("candidate_top_n", 10)
        params.setdefault("deterministic_switch_decision", False)
        params.setdefault("deterministic_pair_selection", False)
        params.setdefault("train_candidate_filter_enabled", True)
        params.setdefault("train_candidate_pool_size", 30)
        params.setdefault("initial_pair_rule", "train_shape")
        super().__init__(pairs=pairs, params=params)
        self.asset_label = str(self.params.get("asset_label", asset_label)).upper()
        self.target_vol = float(self.params.get("target_vol", 0.35))
        self.ann_factor = float(self.params.get("ann_factor", 365.0))
        self.train_candidate_filter_enabled = bool(
            self.params.get("train_candidate_filter_enabled", True)
        )
        self.train_candidate_pool_size = max(
            int(self.params.get("train_candidate_pool_size", 30)),
            self.candidate_top_n,
        )
        self.initial_pair_rule = str(self.params.get("initial_pair_rule", "train_shape")).lower()

    def _train_shape_rows(self, performance_metrics: dict[str, Any]) -> list[tuple[float, str]]:
        rows = performance_metrics.get("deterministic_pair_ranking", [])
        if not isinstance(rows, list):
            return []
        scored: list[tuple[float, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("pair", ""))
            if not name:
                continue
            sharpe = self._safe_float(row.get("sharpe"), 0.0)
            drawdown = abs(self._safe_float(row.get("drawdown"), 0.0))
            turnover = self._safe_float(row.get("turnover"), 0.0)
            realized_vol = self._safe_float(row.get("realized_vol"), 0.0)
            vol_error = self._safe_float(
                row.get("vol_tracking_error"),
                abs(realized_vol - self.target_vol),
            )
            score = (
                sharpe
                - 0.75 * drawdown
                - 0.05 * turnover
                - 0.15 * (vol_error / max(self.target_vol, 1e-12))
            )
            scored.append((float(score), name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored

    def _set_initial_pair_from_train_shape(self, performance_metrics: dict[str, Any]) -> None:
        if self._active_pair is not None or self.initial_pair_rule != "train_shape":
            return
        for _, name in self._train_shape_rows(performance_metrics):
            pair = next((candidate for candidate in self.pairs if candidate.name == name), None)
            if pair is not None:
                self.default_pair = pair
                return

    def _train_candidate_names(self, performance_metrics: dict[str, Any]) -> set[str]:
        if not self.train_candidate_filter_enabled:
            return set()
        return {name for _, name in self._train_shape_rows(performance_metrics)[: self.train_candidate_pool_size]}

    def _regime_pair_pool(
        self,
        regime: str,
        performance_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidates = super()._regime_pair_pool(regime, performance_metrics)
        allowed_names = self._train_candidate_names(performance_metrics)
        if not allowed_names:
            return candidates
        filtered = [row for row in candidates if str(row.get("name")) in allowed_names]
        return filtered or candidates

    def _btc_metric_view(self, candidate: dict[str, Any]) -> dict[str, float]:
        recent = candidate.get("recent_performance") or {}
        overall = candidate.get("overall_rank_context") or {}
        regime = candidate.get("regime_history") or {}
        active_regime = candidate.get("active_regime_ranks") or {}

        realized_vol = self._safe_float(
            recent.get(
                "realized_vol",
                active_regime.get("realized_vol", regime.get("realized_vol", overall.get("realized_vol"))),
            ),
            0.0,
        )
        vol_error = self._safe_float(
            recent.get(
                "vol_tracking_error",
                active_regime.get(
                    "vol_tracking_error",
                    regime.get("vol_tracking_error", overall.get("vol_tracking_error")),
                ),
            ),
            abs(realized_vol - self.target_vol),
        )
        sharpe = self._safe_float(
            recent.get("rolling_sharpe", active_regime.get("sharpe", regime.get("sharpe", overall.get("sharpe")))),
            0.0,
        )
        drawdown = abs(
            self._safe_float(
                recent.get(
                    "drawdown",
                    active_regime.get("max_drawdown", regime.get("max_drawdown", overall.get("drawdown"))),
                ),
                0.0,
            )
        )
        turnover = self._safe_float(
            recent.get("turnover", regime.get("avg_turnover", overall.get("turnover"))),
            0.0,
        )
        cvar = abs(self._safe_float(recent.get("cvar_95", recent.get("tail_loss", 0.0)), 0.0))
        return {
            "realized_vol": realized_vol,
            "vol_tracking_error": vol_error,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "turnover": turnover,
            "cvar_95": cvar,
        }

    def _sort_rows_by_turnover_drawdown_sharpe(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                self._safe_float(row.get("vol_tracking_error"), abs(self._safe_float(row.get("realized_vol"), 0.0) - self.target_vol)),
                -self._safe_float(row.get("sharpe"), 0.0),
                self._safe_float(row.get("max_drawdown"), 0.0),
                self._safe_float(row.get("avg_turnover"), 0.0),
                str(row.get("pair", "")),
            ),
        )

    def _candidate_selection_rank(self, candidate: dict[str, Any]) -> tuple[int, str]:
        metrics = self._btc_metric_view(candidate)
        vol_bucket = int(metrics["vol_tracking_error"] * 1_000_000)
        return (
            vol_bucket,
            f"{-metrics['sharpe']:020.12f}:{metrics['drawdown']:020.12f}:{metrics['turnover']:020.12f}:{candidate.get('name', '')}",
        )

    def _candidate_metric_score(self, candidate: dict[str, Any]) -> float:
        metrics = self._btc_metric_view(candidate)
        benchmark_edge = candidate.get("vs_rv22_naive_scaling") or {}
        sharpe_edge = self._safe_float(benchmark_edge.get("rolling_sharpe_minus_benchmark"), 0.0)
        vol_penalty = metrics["vol_tracking_error"]
        under_target_penalty = max(self.target_vol * 0.70 - metrics["realized_vol"], 0.0)
        over_target_penalty = max(metrics["realized_vol"] - self.target_vol * 1.30, 0.0)
        return float(
            1.35 * metrics["sharpe"]
            + 0.45 * sharpe_edge
            - 4.00 * vol_penalty
            - 2.00 * under_target_penalty
            - 2.50 * over_target_penalty
            - 1.75 * metrics["drawdown"]
            - 0.75 * metrics["cvar_95"]
            - 0.10 * metrics["turnover"]
        )

    def _selection_payload(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> str:
        payload = json.loads(super()._selection_payload(market_features, performance_metrics, candidates))
        payload["crypto_objective"] = {
            "target_vol": self.target_vol,
            "ann_factor": self.ann_factor,
            "baseline": "20d realized-vol + naive scaling",
            "primary_rule": (
                f"Choose a non-active pair that improves Sharpe while keeping realized_vol near {self.target_vol:.6f}. "
                "The supplied pairs are filtered using only train-window and prior-day candidate metrics. "
                "Do not choose a very low-volatility pair unless all target-vol candidates have bad Sharpe/drawdown."
            ),
        }
        payload["instruction"] = (
            f"Choose the best supplied {self.asset_label} pair for the next checkpoint. "
            "Primary objective is Sharpe better than the 20d realized-vol naive baseline while tracking "
            f"the {_pct(self.target_vol)} annualized volatility target using ann_factor={self.ann_factor:.0f}. "
            "Rank candidates by vol_tracking_error first, then Sharpe, drawdown, benchmark Sharpe edge, and turnover. "
            "Return only {\"pair\":\"<supplied pair name>\"}."
        )
        return json.dumps(payload, default=str)

    def _switch_decision_payload(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> str:
        payload = json.loads(super()._switch_decision_payload(market_features, performance_metrics))
        active_perf = payload.get("active_pair_performance", {})
        active_vol = self._safe_float(active_perf.get("realized_vol"), 0.0) if isinstance(active_perf, dict) else 0.0
        payload["crypto_objective"] = {
            "target_vol": self.target_vol,
            "ann_factor": self.ann_factor,
            "active_realized_vol": active_vol,
            "active_vol_tracking_error": abs(active_vol - self.target_vol) if active_vol > 0 else None,
            "switch_bias": (
                f"Switch if active realized volatility is far from {self.target_vol:.6f}, "
                f"especially below {self.target_vol * 0.70:.6f} or above {self.target_vol * 1.30:.6f}, "
                "and a challenger has competitive Sharpe/drawdown."
            ),
        }
        payload["instruction"] = (
            "Return only {\"action\":\"hold\"} or {\"action\":\"switch\"}. "
            f"This {self.asset_label} strategy targets {_pct(self.target_vol)} annualized realized volatility "
            f"with ann_factor={self.ann_factor:.0f}. "
            f"Favor switch when active vol tracking is poor, active recent Sharpe lags challengers, or {self.asset_label} regime changed. "
            "Favor hold only when active vol is near target and active recent ranks remain competitive."
        )
        return json.dumps(payload, default=str)

    def select(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        timestamp: Any = None,
    ):
        self._set_initial_pair_from_train_shape(performance_metrics)
        chosen = super().select(
            market_features=market_features,
            diagnostics=diagnostics,
            performance_metrics=performance_metrics,
            timestamp=timestamp,
        )
        if self.decisions:
            self.decisions[-1]["btc_target_vol"] = self.target_vol
            self.decisions[-1]["btc_ann_factor"] = self.ann_factor
        return chosen


# Backward-compatible names for existing configs that import this module path.
BTCVolTargetAIRouter = CryptoVolTargetAIRouter
BTCBenchmarkGuardRouter = CryptoVolTargetAIRouter


__all__ = ["CryptoVolTargetAIRouter", "BTCVolTargetAIRouter", "BTCBenchmarkGuardRouter"]
