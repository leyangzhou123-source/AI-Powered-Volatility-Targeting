"""LLM-backed router for choosing estimator-controller pairs.

The model is allowed to choose only one of the configured pair names. If the API
key is empty, the request fails, or the response is invalid, the router falls
back to the transparent base score from ``Router``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.router.router import Router
from src.router.strategy_pair import StrategyPair


SYSTEM_PROMPT = """You are a conservative routing agent for a volatility-targeting backtest.
Choose exactly one estimator-controller pair from the provided candidate names.
Prefer robust risk control, low switching, low invalid/exception rates, and good
recent risk-adjusted performance. Return strict JSON only:
{"pair": "<candidate name>", "reason": "<short reason>", "confidence": 0.0}
Keep reason under 24 words.
"""


DECISION_SYSTEM_PROMPT = """Output JSON immediately. Do not explain. Do not reason step by step.
You are the review layer for a volatility-targeting router.
You may see only past intraday volatility, strategy return/drawdown/volatility, benchmark performance, active pair state, and switch frequency.
Decide whether to hold the current estimator-controller pair or request a switch review.
Penalize frequent switching heavily. Request switch only when return/drawdown/benchmark evidence or volatility stream persistence justifies it.
Return only this strict JSON object:
{"action": "hold", "reason": "<short reason>", "confidence": 0.0}
or:
{"action": "switch", "reason": "<short reason>", "confidence": 0.0}
Keep reason under 24 words.
"""


class LLMRouter(Router):
    """Selects a strategy pair by asking an LLM, with deterministic fallback."""

    responses_endpoint = "https://api.openai.com/v1/responses"
    chat_completions_endpoint = "https://api.openai.com/v1/chat/completions"
    nvidia_endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, pairs: list[StrategyPair], params: dict[str, Any] | None = None):
        super().__init__(pairs, params)
        p = self.params
        self.provider = str(p.get("provider", "openai")).lower()
        self.api_format = str(p.get("api_format", "responses")).lower()
        self.model = str(p.get("model", "gpt-5-mini"))
        self.api_key = str(
            p.get("api_key", "")
            or os.getenv("OPENAI_API_KEY", "")
            or os.getenv("NVIDIA_API_KEY", "")
            or os.getenv("NVAPI_KEY", "")
        )
        self.endpoint = str(p.get("endpoint", "") or self._default_endpoint())
        self.system_prompt = str(p.get("system_prompt", SYSTEM_PROMPT))
        self.max_output_tokens = int(p.get("max_output_tokens", 512))
        self.temperature = float(p.get("temperature", 0.0))
        self.reasoning_effort = p.get("reasoning_effort")
        self.timeout = float(p.get("timeout", 30.0))
        self.response_format = p.get("response_format")
        self.raw_response_debug_dir = str(p.get("raw_response_debug_dir", "results/evaluation/llm_raw_responses"))
        self.retry_on_invalid_json = bool(p.get("retry_on_invalid_json", True))
        self.retry_candidate_top_n = int(p.get("retry_candidate_top_n", 10))
        self.max_calls = int(p.get("max_calls", 0))
        self.two_stage_decision = bool(p.get("two_stage_decision", False))
        self.review_interval = max(int(p.get("review_interval", 20)), 1)
        self.decision_system_prompt = str(p.get("decision_system_prompt", DECISION_SYSTEM_PROMPT))
        self.selection_system_prompt = str(p.get("selection_system_prompt", self.system_prompt))
        self.llm_fallback_score_prompt_weight = float(p.get("llm_fallback_score_prompt_weight", 0.15))
        self.llm_regime_suitability_prompt_scale = float(p.get("llm_regime_suitability_prompt_scale", 0.25))
        self.turnover_penalty_prompt_weight = float(p.get("turnover_penalty_prompt_weight", 4.0))
        self.llm_drawdown_penalty_prompt_weight = float(p.get("llm_drawdown_penalty_prompt_weight", 0.5))
        self.llm_vol_band_penalty_prompt_weight = float(p.get("llm_vol_band_penalty_prompt_weight", 2.0))
        self.llm_return_reward_prompt_weight = float(p.get("llm_return_reward_prompt_weight", 1.0))
        self.llm_target_vol = float(p.get("llm_target_vol", 0.10))
        self.llm_vol_min = float(p.get("llm_vol_min", 0.09))
        self.llm_vol_max = float(p.get("llm_vol_max", 0.11))
        self.llm_max_pair_drawdown = float(p.get("llm_max_pair_drawdown", 0.08))
        self.candidate_hard_risk_filter = bool(p.get("candidate_hard_risk_filter", False))
        self.candidate_risk_filter_min_count = int(p.get("candidate_risk_filter_min_count", 10))
        self.llm_hard_vol_max = float(p.get("llm_hard_vol_max", 0.12))
        self.llm_hard_drawdown_max = float(p.get("llm_hard_drawdown_max", 0.12))
        self.benchmark_required_margin = float(p.get("benchmark_required_margin", 0.0))
        self.switch_frequency_penalty_weight = float(p.get("switch_frequency_penalty_weight", 2.0))
        self.pair_concentration_window = int(p.get("pair_concentration_window", 252))
        self.pair_concentration_threshold = float(p.get("pair_concentration_threshold", 0.70))
        self.decision_mode = str(p.get("decision_mode", "interval")).lower()
        self.decision_interval = max(int(p.get("decision_interval", 1)), 1)
        self.min_decision_gap = max(int(p.get("min_decision_gap", self.decision_interval)), 1)
        self.rv_zscore_trigger = float(p.get("rv_zscore_trigger", 1.5))
        self.rv_change_trigger = float(p.get("rv_change_trigger", 0.25))
        self.rv_percentile_trigger = float(p.get("rv_percentile_trigger", 0.80))
        self.strategy_drawdown_trigger = float(p.get("strategy_drawdown_trigger", 0.08))
        self.active_pair_drawdown_trigger = float(p.get("active_pair_drawdown_trigger", 0.06))
        self.benchmark_underperformance_trigger = float(p.get("benchmark_underperformance_trigger", 0.02))
        self.performance_event_lookback = str(p.get("performance_event_lookback", "trailing_63d"))
        self.decision_recent_lookback = str(p.get("decision_recent_lookback", "trailing_10d"))
        raw_prompt_lookbacks = p.get("prompt_history_lookbacks", [self.performance_event_lookback])
        if isinstance(raw_prompt_lookbacks, str):
            self.prompt_history_lookbacks = [raw_prompt_lookbacks]
        else:
            self.prompt_history_lookbacks = [str(x) for x in raw_prompt_lookbacks]
        self.always_call_first = bool(p.get("always_call_first", True))
        self.include_scores = bool(p.get("include_scores", True))
        self.include_regime_suitability_prompt = bool(p.get("include_regime_suitability_prompt", True))
        self.fail_open = bool(p.get("fail_open", True))
        self.max_consecutive_pair_calls = int(p.get("max_consecutive_pair_calls", 0))
        self.diversity_score_margin = float(p.get("diversity_score_margin", 0.25))
        self.recent_choice_window = int(p.get("recent_choice_window", 10))
        self.candidate_top_n = int(p.get("candidate_top_n", 0))
        self.candidate_rank_mode = str(p.get("candidate_rank_mode", "fallback")).lower()

        self._call_count = 0
        self._attempt_count = 0
        self._step = 0
        self._last_llm_step: int | None = None
        self._recent_llm_choices: list[str] = []
        self._last_raw_response: dict[str, Any] | None = None
        self._switch_history: list[int] = []
        self._selected_pair_history: list[str] = []

    def _compact_trailing_performance(
        self,
        value: Any,
        lookbacks: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        if not isinstance(value, dict):
            return {}
        out: dict[str, dict[str, float]] = {}
        for lookback in (lookbacks or self.prompt_history_lookbacks):
            metrics = value.get(lookback)
            if not isinstance(metrics, dict):
                continue
            out[lookback] = {
                "trailing_return": round(self._safe_float(metrics.get("trailing_return")), 6),
                "annualized_return": round(self._safe_float(metrics.get("annualized_return")), 6),
                "rolling_sharpe": round(self._safe_float(metrics.get("rolling_sharpe")), 6),
                "drawdown": round(self._safe_float(metrics.get("drawdown")), 6),
                "realized_vol": round(self._safe_float(metrics.get("realized_vol")), 6),
                "vol_band_error": round(self._safe_float(metrics.get("vol_band_error")), 6),
                "turnover": round(self._safe_float(metrics.get("turnover")), 6),
            }
        return out

    def _compact_benchmark_metrics(self, performance_metrics: dict[str, Any]) -> dict[str, Any]:
        benchmark = performance_metrics.get("benchmark")
        if isinstance(benchmark, dict):
            return self._compact_trailing_performance(benchmark)
        return {}

    def _strategy_performance_for_prompt(self, performance_metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            key: round(self._safe_float(performance_metrics.get(key)), 6)
            for key in (
                "obs",
                "rolling_sharpe",
                "drawdown",
                "realized_vol",
                "vol_tracking_error",
                "trailing_10d_return",
                "trailing_63d_return",
                "trailing_126d_return",
            )
            if key in performance_metrics
        }

    def _recent_10d_decision_focus(
        self,
        active_perf: Any,
        performance_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        lookback = self.decision_recent_lookback
        active = self._compact_trailing_performance(active_perf, [lookback]).get(lookback, {})
        benchmark = self._compact_trailing_performance(performance_metrics.get("benchmark"), [lookback]).get(
            lookback,
            {},
        )
        active_return = self._safe_float(active.get("trailing_return"), 0.0)
        benchmark_return = self._safe_float(benchmark.get("trailing_return"), 0.0)
        return {
            "lookback": lookback,
            "active_pair": active,
            "benchmark": benchmark,
            "return_minus_benchmark": round(active_return - benchmark_return, 6),
            "instruction": (
                "Use this recent window first. Switch only if recent evidence plus "
                "longer context supports overcoming turnover and switch-frequency cost."
            ),
        }

    def _risk_adjusted_candidate_score(
        self,
        pair: StrategyPair,
        performance_metrics: dict[str, Any],
        scores: dict[str, float],
    ) -> float:
        if self.candidate_rank_mode not in ("risk_adjusted", "risk", "llm_risk", "performance_first"):
            return self._safe_float(scores.get(pair.name, float("-inf")))

        pair_perf = performance_metrics.get(pair.name)
        benchmark_perf = performance_metrics.get("benchmark")
        compact_perf = self._compact_trailing_performance(pair_perf)
        compact_benchmark = self._compact_trailing_performance(benchmark_perf)
        if not compact_perf:
            if self.candidate_rank_mode == "performance_first":
                return float("-inf")
            return self._safe_float(scores.get(pair.name, 0.0)) * self.llm_fallback_score_prompt_weight

        return_edges = []
        turnovers = []
        drawdowns = []
        vol_errors = []
        for lookback, metrics in compact_perf.items():
            bench = compact_benchmark.get(lookback, {})
            return_edges.append(
                self._safe_float(metrics.get("trailing_return"), 0.0)
                - self._safe_float(bench.get("trailing_return"), 0.0)
            )
            turnovers.append(self._safe_float(metrics.get("turnover"), 0.0))
            drawdowns.append(self._safe_float(metrics.get("drawdown"), 0.0))
            realized_vol = self._safe_float(metrics.get("realized_vol"), self.llm_target_vol)
            vol_errors.append(abs(realized_vol - self.llm_target_vol))

        return_edge = sum(return_edges) / max(len(return_edges), 1)
        turnover = max(turnovers) if turnovers else 0.0
        drawdown = max(drawdowns) if drawdowns else 0.0
        vol_error = max(vol_errors) if vol_errors else 0.0
        fallback = 0.0
        if self.candidate_rank_mode != "performance_first":
            fallback = self._safe_float(scores.get(pair.name, 0.0)) * self.llm_fallback_score_prompt_weight
        return (
            self.llm_return_reward_prompt_weight * return_edge
            + fallback
            - self.turnover_penalty_prompt_weight * turnover
            - self.llm_drawdown_penalty_prompt_weight * drawdown
            - self.llm_vol_band_penalty_prompt_weight * vol_error
        )

    def _candidate_risk_check(self, pair: StrategyPair, performance_metrics: dict[str, Any]) -> dict[str, Any]:
        compact_perf = self._compact_trailing_performance(performance_metrics.get(pair.name))
        if not compact_perf:
            return {
                "has_history": False,
                "passes_hard_filter": True,
                "max_realized_vol": 0.0,
                "max_drawdown": 0.0,
            }
        realized_vols = [
            self._safe_float(metrics.get("realized_vol"), 0.0)
            for metrics in compact_perf.values()
            if isinstance(metrics, dict)
        ]
        drawdowns = [
            self._safe_float(metrics.get("drawdown"), 0.0)
            for metrics in compact_perf.values()
            if isinstance(metrics, dict)
        ]
        max_vol = max(realized_vols) if realized_vols else 0.0
        max_drawdown = max(drawdowns) if drawdowns else 0.0
        return {
            "has_history": True,
            "passes_hard_filter": bool(
                max_vol <= self.llm_hard_vol_max
                and max_drawdown <= self.llm_hard_drawdown_max
            ),
            "max_realized_vol": round(max_vol, 6),
            "max_drawdown": round(max_drawdown, 6),
            "hard_vol_max": round(self.llm_hard_vol_max, 6),
            "hard_drawdown_max": round(self.llm_hard_drawdown_max, 6),
        }

    def _recent_switch_frequency(self, window: int = 126) -> float:
        if not self._switch_history:
            return 0.0
        recent = self._switch_history[-window:]
        return float(sum(recent) / max(len(recent), 1))

    def _pair_concentration_stats(self) -> dict[str, Any]:
        if not self._selected_pair_history:
            return {
                "window": self.pair_concentration_window,
                "dominant_pair": self._active_pair.name if self._active_pair else None,
                "dominant_share": 0.0,
                "threshold": self.pair_concentration_threshold,
                "is_concentrated": False,
            }
        recent = self._selected_pair_history[-self.pair_concentration_window :]
        counts: dict[str, int] = {}
        for name in recent:
            counts[name] = counts.get(name, 0) + 1
        dominant_pair, count = max(counts.items(), key=lambda kv: kv[1])
        share = float(count / max(len(recent), 1))
        return {
            "window": self.pair_concentration_window,
            "dominant_pair": dominant_pair,
            "dominant_share": round(share, 6),
            "threshold": self.pair_concentration_threshold,
            "is_concentrated": bool(share >= self.pair_concentration_threshold),
        }

    def _save_raw_response(self, error: str, prompt_kind: str) -> str:
        if self._last_raw_response is None:
            return ""
        try:
            out_dir = Path(self.raw_response_debug_dir)
            if not out_dir.is_absolute():
                out_dir = Path.cwd() / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"llm_step_{self._step:06d}_call_{self._call_count:03d}_{prompt_kind}.json"
            payload = {
                "step": self._step,
                "call_count": self._call_count,
                "model": self.model,
                "provider": self.provider,
                "endpoint": self.endpoint,
                "error": error,
                "raw_response": self._last_raw_response,
            }
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return str(path)
        except Exception:
            return ""

    def _default_endpoint(self) -> str:
        if self.provider == "nvidia":
            return self.nvidia_endpoint
        if self.api_format in ("chat", "chat_completions", "chat-completions"):
            return self.chat_completions_endpoint
        return self.responses_endpoint

    def _fallback_pair(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> tuple[StrategyPair, dict[str, float], dict[str, dict[str, float]]]:
        scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        best_pair = self._active_pair or self.default_pair
        best_score = float("-inf")

        for pair in self.pairs:
            c = super()._score_components(pair, market_features, diagnostics, performance_metrics)
            score = float(c["total"])
            scores[pair.name] = score
            components[pair.name] = c
            if score > best_score:
                best_score = score
                best_pair = pair

        return best_pair, scores, components

    def _candidate_payload(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        scores: dict[str, float],
        exclude_pair_names: set[str] | None = None,
    ) -> str:
        excluded = set(exclude_pair_names or set())
        pairs_for_prompt = [pair for pair in self.pairs if pair.name not in excluded]
        risk_filter_applied = False
        risk_filtered_count = 0
        if self.candidate_hard_risk_filter:
            passing = [
                pair for pair in pairs_for_prompt
                if self._candidate_risk_check(pair, performance_metrics)["passes_hard_filter"]
            ]
            if len(passing) >= max(self.candidate_risk_filter_min_count, 1):
                risk_filtered_count = len(pairs_for_prompt) - len(passing)
                pairs_for_prompt = passing
                risk_filter_applied = True
        if self.candidate_top_n > 0 and len(pairs_for_prompt) > self.candidate_top_n:
            ranked = sorted(
                pairs_for_prompt,
                key=lambda pair: self._risk_adjusted_candidate_score(pair, performance_metrics, scores),
                reverse=True,
            )
            keep: dict[str, StrategyPair] = {pair.name: pair for pair in ranked[: self.candidate_top_n]}
            if self._active_pair is not None and self._active_pair.name not in excluded:
                keep[self._active_pair.name] = self._active_pair
            for recent in self._recent_llm_choices[-self.recent_choice_window :]:
                if recent in excluded:
                    continue
                pair = next((candidate for candidate in self.pairs if candidate.name == recent), None)
                if pair is not None:
                    keep[pair.name] = pair
            pairs_for_prompt = list(keep.values())

        candidates = []
        for pair in pairs_for_prompt:
            pair_diag = self._pair_diagnostics(pair, diagnostics)
            regime_components = self._regime_bias_components(pair, market_features)
            hard_risk_check = self._candidate_risk_check(pair, performance_metrics)
            scaled_regime = {
                "pair": round(float(regime_components["pair_regime_bias"]) * self.llm_regime_suitability_prompt_scale, 6),
                "estimator": round(float(regime_components["estimator_regime_bias"]) * self.llm_regime_suitability_prompt_scale, 6),
                "controller": round(float(regime_components["controller_regime_bias"]) * self.llm_regime_suitability_prompt_scale, 6),
                "heuristic": round(float(regime_components["heuristic_regime_bias"]) * self.llm_regime_suitability_prompt_scale, 6),
                "total": round(float(regime_components["total"]) * self.llm_regime_suitability_prompt_scale, 6),
            }
            row = {
                "name": pair.name,
                "estimator": pair.estimator_name,
                "controller": pair.controller_name,
                "hard_risk_filter_check": hard_risk_check,
            }
            if self.include_regime_suitability_prompt:
                row["regime_suitability_weak"] = scaled_regime
            if self.include_scores:
                row["fallback_score_weak"] = round(
                    float(scores.get(pair.name, 0.0)) * self.llm_fallback_score_prompt_weight,
                    6,
                )
            pair_perf = performance_metrics.get(pair.name)
            if isinstance(pair_perf, dict):
                compact_perf = self._compact_trailing_performance(pair_perf)
                if compact_perf:
                    row["trailing_performance"] = compact_perf
                    benchmark_perf = self._compact_benchmark_metrics(performance_metrics)
                    lookback = self.prompt_history_lookbacks[0] if self.prompt_history_lookbacks else self.performance_event_lookback
                    pair_window = compact_perf.get(lookback, {})
                    bench_window = benchmark_perf.get(lookback, {})
                    pair_ret = self._safe_float(pair_window.get("trailing_return"), 0.0)
                    bench_ret = self._safe_float(bench_window.get("trailing_return"), 0.0)
                    turnover = self._safe_float(pair_window.get("turnover"), 0.0)
                    drawdown = self._safe_float(pair_window.get("drawdown"), 0.0)
                    realized_vol = self._safe_float(pair_window.get("realized_vol"), self.llm_target_vol)
                    vol_band_error = abs(realized_vol - self.llm_target_vol)
                    risk_drawdowns = [
                        self._safe_float(metrics.get("drawdown"), 0.0)
                        for metrics in compact_perf.values()
                        if isinstance(metrics, dict)
                    ]
                    risk_vol_errors = [
                        abs(self._safe_float(metrics.get("realized_vol"), self.llm_target_vol) - self.llm_target_vol)
                        for metrics in compact_perf.values()
                        if isinstance(metrics, dict)
                    ]
                    max_drawdown = max(risk_drawdowns) if risk_drawdowns else drawdown
                    max_vol_error = max(risk_vol_errors) if risk_vol_errors else vol_band_error
                    row["benchmark_comparison"] = {
                        "benchmark_trailing_return": round(bench_ret, 6),
                        "return_minus_benchmark": round(pair_ret - bench_ret, 6),
                        "meets_benchmark": pair_ret >= bench_ret + self.benchmark_required_margin,
                    }
                    row["risk_target_check"] = {
                        "target_vol": round(self.llm_target_vol, 6),
                        "vol_min": round(self.llm_vol_min, 6),
                        "vol_max": round(self.llm_vol_max, 6),
                        "primary_realized_vol": round(realized_vol, 6),
                        "primary_vol_error": round(vol_band_error, 6),
                        "max_prompt_drawdown": round(max_drawdown, 6),
                        "max_prompt_vol_error": round(max_vol_error, 6),
                        "passes_primary_vol_band": self.llm_vol_min <= realized_vol <= self.llm_vol_max,
                        "passes_drawdown_limit": max_drawdown <= self.llm_max_pair_drawdown,
                    }
                    row["turnover_penalty"] = round(turnover * self.turnover_penalty_prompt_weight, 6)
                    row["llm_net_return_score"] = round(
                        self.llm_return_reward_prompt_weight * (pair_ret
                        - bench_ret
                        )
                        - self.turnover_penalty_prompt_weight * turnover
                        - self.llm_drawdown_penalty_prompt_weight * max_drawdown
                        - self.llm_vol_band_penalty_prompt_weight * max_vol_error,
                        6,
                    )
            if pair_diag:
                row["diagnostics"] = {
                    "invalid_rate": self._safe_float(pair_diag.get("invalid_rate", 0.0)),
                    "exception_rate": self._safe_float(pair_diag.get("exception_rate", 0.0)),
                    "turnover": self._safe_float(pair_diag.get("turnover", 0.0)),
                    "vol_tracking_error": self._safe_float(pair_diag.get("vol_tracking_error", 0.0)),
                    "estimator_loss": self._safe_float(pair_diag.get("estimator_loss", 0.0)),
                }
            candidates.append(row)

        global_performance = {
            key: value
            for key, value in performance_metrics.items()
            if not isinstance(value, dict)
        }
        benchmark = performance_metrics.get("benchmark")
        if isinstance(benchmark, dict):
            global_performance["benchmark"] = self._compact_trailing_performance(benchmark)

        payload = {
            "instruction": (
                "Choose from candidate trailing_performance, benchmark_comparison, "
                "risk_target_check, turnover_penalty, and llm_net_return_score only. "
                "First minimize turnover and drawdown. Second, prefer higher return versus "
                "benchmark after turnover. Keep realized volatility close to "
                f"{self.llm_target_vol:.0%} inside {self.llm_vol_min:.0%}-{self.llm_vol_max:.0%} "
                "as a risk constraint. Avoid letting one pair dominate unless it clearly beats "
                "alternatives on turnover, drawdown, benchmark-relative return, and volatility fit."
            ),
            "active_pair": self._active_pair.name if self._active_pair else None,
            "recent_llm_choices": self._recent_llm_choices[-self.recent_choice_window :],
            "pair_concentration": self._pair_concentration_stats(),
            "total_candidate_count": len(self.pairs),
            "shown_candidate_count": len(candidates),
            "excluded_candidate_names": sorted(excluded),
            "candidate_filter": (
                f"top_{self.candidate_top_n}_by_{self.candidate_rank_mode}_score_plus_active_recent"
                if self.candidate_top_n > 0 else "all"
            ),
            "hard_risk_filter": {
                "enabled": self.candidate_hard_risk_filter,
                "applied": risk_filter_applied,
                "filtered_count": risk_filtered_count,
                "min_remaining": self.candidate_risk_filter_min_count,
                "hard_vol_max": self.llm_hard_vol_max,
                "hard_drawdown_max": self.llm_hard_drawdown_max,
            },
            "market_features": market_features,
            "performance_metrics": global_performance,
            "candidates": candidates,
        }
        return json.dumps(payload, default=str)

    def _decision_payload(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> str:
        active_name = self._active_pair.name if self._active_pair is not None else None
        active_perf = performance_metrics.get(active_name) if active_name else None
        payload = {
            "instruction": (
                "Decision layer only: decide hold or switch_review. "
                "You cannot see candidate list. Penalize switch frequency heavily."
            ),
            "active_pair": active_name,
            "days_since_last_llm_review": (
                self._step if self._last_llm_step is None else self._step - self._last_llm_step
            ),
            "review_interval": self.review_interval,
            "recent_switch_frequency_126d": round(self._recent_switch_frequency(126), 6),
            "switch_frequency_penalty_weight": self.switch_frequency_penalty_weight,
            "pair_concentration": self._pair_concentration_stats(),
            "intraday_realized_vol": market_features.get("intraday_realized_vol", {}),
            "strategy_performance": self._strategy_performance_for_prompt(performance_metrics),
            "recent_10d_focus": self._recent_10d_decision_focus(active_perf, performance_metrics),
            "active_pair_return_info": self._compact_trailing_performance(active_perf),
            "benchmark": self._compact_benchmark_metrics(performance_metrics),
        }
        return json.dumps(payload, default=str)

    def _retry_candidate_payload(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
        scores: dict[str, float],
        exclude_pair_names: set[str] | None = None,
    ) -> str:
        excluded = set(exclude_pair_names or set())
        ranked = sorted(
            [pair for pair in self.pairs if pair.name not in excluded],
            key=lambda pair: self._risk_adjusted_candidate_score(pair, performance_metrics, scores),
            reverse=True,
        )
        keep: dict[str, StrategyPair] = {pair.name: pair for pair in ranked[: max(self.retry_candidate_top_n, 1)]}
        if self._active_pair is not None and self._active_pair.name not in excluded:
            keep[self._active_pair.name] = self._active_pair
        candidates = []
        for pair in keep.values():
            row = {"name": pair.name}
            if self.include_scores:
                row["fallback_score"] = round(float(scores.get(pair.name, 0.0)), 6)
            pair_perf = performance_metrics.get(pair.name)
            compact_perf = self._compact_trailing_performance(pair_perf)
            if compact_perf:
                row["trailing_performance"] = compact_perf
            candidates.append(row)

        global_performance = {
            key: value
            for key, value in performance_metrics.items()
            if not isinstance(value, dict)
        }
        benchmark = performance_metrics.get("benchmark")
        if isinstance(benchmark, dict):
            global_performance["benchmark"] = self._compact_trailing_performance(benchmark)

        payload = {
            "instruction": "Retry with compact payload. Return JSON only with pair, reason, confidence.",
            "active_pair": self._active_pair.name if self._active_pair else None,
            "excluded_candidate_names": sorted(excluded),
            "market_features": {
                "vol_regime": market_features.get("vol_regime"),
                "rolling_vol": market_features.get("rolling_vol"),
                "intraday_realized_vol": market_features.get("intraday_realized_vol"),
            },
            "performance_metrics": global_performance,
            "candidates": candidates,
        }
        return json.dumps(payload, default=str)

    def _intraday_rv_event_reason(self, market_features: dict[str, Any]) -> str | None:
        rv = market_features.get("intraday_realized_vol", {})
        if not isinstance(rv, dict) or not rv:
            return None

        zscore = abs(self._safe_float(rv.get("zscore"), 0.0))
        pct_change_1d = abs(self._safe_float(rv.get("pct_change_1d"), 0.0))
        percentile = self._safe_float(rv.get("percentile"), 0.0)

        reasons = []
        if zscore >= self.rv_zscore_trigger:
            reasons.append(f"rv_zscore={zscore:.2f}")
        if pct_change_1d >= self.rv_change_trigger:
            reasons.append(f"rv_change_1d={pct_change_1d:.2%}")
        if percentile >= self.rv_percentile_trigger:
            reasons.append(f"rv_percentile={percentile:.2f}")
        if bool(rv.get("regime_changed", False)):
            reasons.append("rv_regime_changed")

        return ", ".join(reasons) if reasons else None

    def _performance_event_reason(self, performance_metrics: dict[str, Any]) -> str | None:
        reasons = []
        strategy_drawdown = self._safe_float(performance_metrics.get("drawdown"), 0.0)
        if strategy_drawdown >= self.strategy_drawdown_trigger:
            reasons.append(f"strategy_drawdown={strategy_drawdown:.2%}")

        active_name = self._active_pair.name if self._active_pair is not None else ""
        active_perf = performance_metrics.get(active_name)
        benchmark_perf = performance_metrics.get("benchmark")
        lookback = self.performance_event_lookback
        if isinstance(active_perf, dict):
            active_window = active_perf.get(lookback)
            if isinstance(active_window, dict):
                active_drawdown = self._safe_float(active_window.get("drawdown"), 0.0)
                if active_drawdown >= self.active_pair_drawdown_trigger:
                    reasons.append(f"active_pair_drawdown={active_drawdown:.2%}")

                if isinstance(benchmark_perf, dict):
                    bench_window = benchmark_perf.get(lookback)
                    if isinstance(bench_window, dict):
                        active_ret = self._safe_float(active_window.get("trailing_return"), 0.0)
                        bench_ret = self._safe_float(bench_window.get("trailing_return"), 0.0)
                        underperf = bench_ret - active_ret
                        if underperf >= self.benchmark_underperformance_trigger:
                            reasons.append(f"benchmark_underperformance={underperf:.2%}")

        return ", ".join(reasons) if reasons else None

    def _performance_metrics_for_log(self, performance_metrics: dict[str, Any]) -> dict[str, Any]:
        logged = {
            key: value
            for key, value in performance_metrics.items()
            if not isinstance(value, dict)
        }
        active_name = self._active_pair.name if self._active_pair is not None else ""
        active_perf = performance_metrics.get(active_name)
        if isinstance(active_perf, dict):
            logged["active_pair"] = {active_name: active_perf}
        benchmark = performance_metrics.get("benchmark")
        if isinstance(benchmark, dict):
            logged["benchmark"] = benchmark
        logged["pair_metric_count"] = sum(1 for value in performance_metrics.values() if isinstance(value, dict))
        return logged

    def _should_call_llm(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> tuple[bool, str]:
        if self.always_call_first and self._attempt_count == 0:
            return True, "first_call"

        if self.max_calls > 0 and self._call_count >= self.max_calls:
            return False, "max_calls_reached"

        if self.decision_mode in ("interval", "fixed"):
            return self._step % self.decision_interval == 0, f"interval_{self.decision_interval}"

        gap = self._step if self._last_llm_step is None else self._step - self._last_llm_step
        if gap < self.min_decision_gap:
            return False, f"min_gap_{self.min_decision_gap}"

        if self.decision_mode in ("intraday_rv_event", "rv_event", "vol_event"):
            reasons = [
                reason for reason in (
                    self._intraday_rv_event_reason(market_features),
                    self._performance_event_reason(performance_metrics),
                )
                if reason
            ]
            reason = "; ".join(reasons)
            return bool(reason), (reason or "no_intraday_rv_event")

        if self.decision_mode in ("intraday_rv_daily", "rv_daily", "daily"):
            return True, "daily_intraday_rv_review"

        return self._step % self.decision_interval == 0, f"unknown_mode_fallback_interval_{self.decision_interval}"

    def _should_review_two_stage(self) -> tuple[bool, str]:
        if self.always_call_first and self._attempt_count == 0:
            return True, "first_review"
        if self.max_calls > 0 and self._call_count >= self.max_calls:
            return False, "max_calls_reached"
        gap = self._step if self._last_llm_step is None else self._step - self._last_llm_step
        if gap < self.review_interval:
            return False, f"review_interval_{self.review_interval}"
        return True, f"scheduled_review_{self.review_interval}"

    def _consecutive_choice_count(self, pair_name: str) -> int:
        count = 0
        for recent in reversed(self._recent_llm_choices):
            if recent != pair_name:
                break
            count += 1
        return count

    def _diversified_alternative(
        self,
        chosen_pair: StrategyPair,
        scores: dict[str, float],
    ) -> tuple[StrategyPair, str | None]:
        if self.max_consecutive_pair_calls <= 0:
            return chosen_pair, None

        repeated = self._consecutive_choice_count(chosen_pair.name)
        if repeated < self.max_consecutive_pair_calls:
            return chosen_pair, None

        chosen_score = self._safe_float(scores.get(chosen_pair.name, 0.0))
        alternatives = [
            pair for pair in self.pairs
            if pair.name != chosen_pair.name
            and self._safe_float(scores.get(pair.name, float("-inf"))) >= chosen_score - self.diversity_score_margin
        ]
        if not alternatives:
            return chosen_pair, None

        best_alt = max(alternatives, key=lambda pair: self._safe_float(scores.get(pair.name, float("-inf"))))
        reason = (
            f"diversity_guard: {chosen_pair.name} was selected "
            f"{repeated} consecutive LLM calls; using close alternative {best_alt.name}"
        )
        return best_alt, reason

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        allow_response_format: bool = True,
        allow_reasoning_effort: bool = True,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("OpenAI API key is empty.")
        if self.max_calls > 0 and self._call_count >= self.max_calls:
            raise RuntimeError("LLM router max_calls limit reached.")

        if self.api_format in ("chat", "chat_completions", "chat-completions") or self.provider == "nvidia":
            request_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt or self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }
            if self.response_format and allow_response_format:
                request_payload["response_format"] = self.response_format
            if self.reasoning_effort and allow_reasoning_effort:
                request_payload["reasoning_effort"] = self.reasoning_effort
        else:
            request_payload = {
                "model": self.model,
                "instructions": system_prompt or self.system_prompt,
                "input": prompt,
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }
            if self.reasoning_effort and allow_reasoning_effort:
                request_payload["reasoning"] = {"effort": self.reasoning_effort}
        body = json.dumps(request_payload).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        self._call_count += 1
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if allow_response_format and self.response_format and "response_format" in detail:
                return self._call_llm(
                    prompt,
                    system_prompt=system_prompt,
                    allow_response_format=False,
                    allow_reasoning_effort=allow_reasoning_effort,
                )
            if allow_reasoning_effort and self.reasoning_effort and "reasoning" in detail:
                return self._call_llm(
                    prompt,
                    system_prompt=system_prompt,
                    allow_response_format=allow_response_format,
                    allow_reasoning_effort=False,
                )
            raise RuntimeError(f"OpenAI API request failed: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

        self._last_raw_response = data
        text = self._extract_text(data)
        try:
            return self._parse_json(text)
        except Exception as exc:
            debug_path = self._save_raw_response(str(exc), "primary")
            if debug_path:
                raise ValueError(f"{exc}; raw_response_saved={debug_path}") from exc
            raise

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            text = choices[0].get("text")
            if isinstance(text, str):
                return text
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str):
                            chunks.append(text)
                    elif isinstance(item, str):
                        chunks.append(item)
                if chunks:
                    return "\n".join(chunks).strip()

        chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()

    def _retry_llm_if_needed(
        self,
        error: str,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
        scores: dict[str, float],
        exclude_pair_names: set[str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        if not self.retry_on_invalid_json:
            return {}, error
        if "not JSON" not in error and "JSON" not in error:
            return {}, error
        if self.max_calls > 0 and self._call_count >= self.max_calls:
            return {}, error

        prompt = self._retry_candidate_payload(market_features, performance_metrics, scores, exclude_pair_names)
        try:
            return self._call_llm(prompt, system_prompt=self.selection_system_prompt), None
        except Exception as exc:
            retry_error = str(exc)
            debug_path = self._save_raw_response(retry_error, "retry")
            if debug_path and "raw_response_saved=" not in retry_error:
                retry_error = f"{retry_error}; raw_response_saved={debug_path}"
            return {}, f"{error}; retry_error={retry_error}"

    def _select_one_stage(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        scores: dict[str, float],
        fallback_pair: StrategyPair,
    ) -> tuple[StrategyPair, bool, dict[str, Any], str | None, str | None, str]:
        llm_response: dict[str, Any] = {}
        error: str | None = None
        diversity_override: str | None = None
        chosen_pair = fallback_pair
        used_llm = False

        should_call, call_reason = self._should_call_llm(market_features, performance_metrics)
        if should_call:
            self._attempt_count += 1
            self._last_llm_step = self._step
            prompt = self._candidate_payload(market_features, diagnostics, performance_metrics, scores)
            try:
                llm_response = self._call_llm(prompt, system_prompt=self.selection_system_prompt)
            except Exception as exc:
                error = str(exc)
                llm_response, error = self._retry_llm_if_needed(error, market_features, performance_metrics, scores)
            try:
                if not llm_response:
                    raise ValueError(error or "LLM response was empty.")
                name = str(llm_response.get("pair", ""))
                candidate = next((pair for pair in self.pairs if pair.name == name), None)
                if candidate is None:
                    raise ValueError(f"LLM selected unknown pair: {name!r}")
                chosen_pair, diversity_override = self._diversified_alternative(candidate, scores)
                used_llm = True
                self._recent_llm_choices.append(candidate.name)
            except Exception as exc:
                error = str(exc) if error is None else error
                if not self.fail_open:
                    raise
        return chosen_pair, used_llm, llm_response, error, diversity_override, call_reason

    def _select_two_stage(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        scores: dict[str, float],
        fallback_pair: StrategyPair,
    ) -> tuple[StrategyPair, bool, dict[str, Any], str | None, str | None, str, dict[str, Any]]:
        chosen_pair = self._active_pair or self.default_pair
        used_llm = False
        llm_response: dict[str, Any] = {}
        decision_response: dict[str, Any] = {}
        error: str | None = None
        diversity_override: str | None = None
        should_review, call_reason = self._should_review_two_stage()
        if not should_review:
            return chosen_pair, used_llm, llm_response, error, diversity_override, call_reason, decision_response

        self._attempt_count += 1
        self._last_llm_step = self._step
        try:
            decision_response = self._call_llm(
                self._decision_payload(market_features, performance_metrics),
                system_prompt=self.decision_system_prompt,
            )
            action = str(decision_response.get("action", "hold")).lower()
            if action not in ("switch", "switch_review", "review"):
                llm_response = {"decision": decision_response, "selection": {}}
                used_llm = True
                return chosen_pair, used_llm, llm_response, error, diversity_override, call_reason, decision_response

            active_name = self._active_pair.name if self._active_pair is not None else ""
            selection_exclusions = {active_name} if active_name else set()
            selection_prompt = self._candidate_payload(
                market_features,
                diagnostics,
                performance_metrics,
                scores,
                exclude_pair_names=selection_exclusions,
            )
            try:
                selection_response = self._call_llm(selection_prompt, system_prompt=self.selection_system_prompt)
            except Exception as exc:
                selection_error = str(exc)
                selection_response, selection_error = self._retry_llm_if_needed(
                    selection_error,
                    market_features,
                    performance_metrics,
                    scores,
                    exclude_pair_names=selection_exclusions,
                )
                if selection_error:
                    raise ValueError(selection_error)

            name = str(selection_response.get("pair", ""))
            if name in selection_exclusions:
                raise ValueError(f"LLM selected excluded active pair during switch: {name!r}")
            candidate = next((pair for pair in self.pairs if pair.name == name), None)
            if candidate is None:
                raise ValueError(f"LLM selected unknown pair: {name!r}")
            chosen_pair, diversity_override = self._diversified_alternative(candidate, scores)
            self._recent_llm_choices.append(candidate.name)
            llm_response = {"decision": decision_response, "selection": selection_response}
            used_llm = True
        except Exception as exc:
            error = str(exc)
            chosen_pair = fallback_pair if self.fail_open else chosen_pair
            if not self.fail_open:
                raise
        return chosen_pair, used_llm, llm_response, error, diversity_override, call_reason, decision_response

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError(f"LLM response was not JSON: {text!r}")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object.")
        return parsed

    def select(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        timestamp: Any = None,
    ) -> StrategyPair:
        if self._active_pair is None:
            self._active_pair = self.default_pair

        fallback_pair, scores, score_components = self._fallback_pair(
            market_features, diagnostics, performance_metrics
        )
        if self.two_stage_decision:
            (
                chosen_pair,
                used_llm,
                llm_response,
                error,
                diversity_override,
                call_reason,
                decision_response,
            ) = self._select_two_stage(
                market_features, diagnostics, performance_metrics, scores, fallback_pair
            )
        else:
            (
                chosen_pair,
                used_llm,
                llm_response,
                error,
                diversity_override,
                call_reason,
            ) = self._select_one_stage(market_features, diagnostics, performance_metrics, scores, fallback_pair)
            decision_response = {}

        can_switch = (self._active_since + 1) >= self.sticky_period
        switched = False
        if can_switch and chosen_pair.name != self._active_pair.name:
            self._active_pair = chosen_pair
            self._active_since = 0
            switched = True
        else:
            self._active_since += 1
        self._switch_history.append(1 if switched else 0)
        self._selected_pair_history.append(self._active_pair.name)

        decision = {
            "timestamp": timestamp,
            "selected_pair": self._active_pair.name,
            "selected_estimator": self._active_pair.estimator_name,
            "selected_controller": self._active_pair.controller_name,
            "switched": switched,
            "scores": scores,
            "score_components": score_components,
            "market_features": dict(market_features),
            "performance_metrics": self._performance_metrics_for_log(performance_metrics),
            "llm_used": used_llm,
            "llm_response": llm_response,
            "llm_decision_response": decision_response,
            "llm_error": error,
            "llm_call_count": self._call_count,
            "llm_attempt_count": self._attempt_count,
            "llm_call_reason": call_reason,
            "llm_diversity_override": diversity_override,
            "recent_llm_choices": list(self._recent_llm_choices[-self.recent_choice_window :]),
        }
        self.decisions.append(decision)
        self._step += 1
        return self._active_pair
