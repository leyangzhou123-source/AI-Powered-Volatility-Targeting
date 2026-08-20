"""AI router for multi-asset portfolio pair selection."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


PORTFOLIO_REGIMES = ("risk_on", "balanced", "defensive")


def load_dotenv(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


AI_PORTFOLIO_REGIME_PROMPT = """Output JSON immediately. Do not explain.
You generate regimes for a combined multi-asset portfolio, not for a single market.
Use the supplied combined portfolio metrics and classify the current portfolio state as one of:
risk_on, balanced, defensive.
Use only past information in the payload. Prefer persistent realized volatility, target-vol error, drawdown,
turnover, pair dispersion, and recent regime history over one-day noise.
Use all three regimes when evidence supports them; keep the regime series diversified and do not collapse most dates into balanced.
risk_on means strong returns/Sharpe with volatility near target and acceptable drawdown.
balanced means moderate return, controlled volatility, and no strong defensive/risk-on signal.
defensive means weaker return, rising drawdown, or volatility slipping outside target.
Return only:
{"portfolio_regime": "balanced", "confidence": 0.0, "reason": "<short reason>"}
Keep reason under 24 words.
"""


AI_PORTFOLIO_SELECTION_PROMPT = """Output JSON immediately. Do not explain.
You are selecting one multi-asset covariance-estimator/controller pair.
Primary objective: select the estimator-controller pair you predict will have the strongest return over the next ten trading days.
Among high-return candidates, prefer the one with better Sharpe and more reliable drawdown control.
Use the supplied 100/60/20-day rankings for each pair. Ranking prioritizes Sharpe first, then drawdown, then turnover, then cVaR.
Use the supplied pair metrics and equity-curve snapshots over the past 100/60/20 days.
Use the supplied drawdown-consistency profile as the main risk check. Consistency means the pair's drawdowns stay stable across windows and close to the supplied target_abs_drawdown, currently about 6%.
Also inspect max_abs_drawdown and max_drawdown_target_gap separately. Among high-return candidates, prefer the one whose worst recent drawdown is controlled and close to the target.
Avoid candidates with unstable, expanding, or materially worse recent/max drawdowns even when their recent return is high.
Use the supplied rankings and statistics across different volatility regimes, plus the recent volatility-regime time series and current regime forecast.
A reasonable approach is to prefer a pair with higher ranking relative to the current regime, but rank must not be the only reason to select it.
Stability matters: avoid unnecessary switching when the current pair remains competitive across recent ranks, equity curves, and regime context.
Choose exactly one supplied non-active pair name. The switch-review layer has already decided not to hold.
Return only:
{"action": "switch", "pair": "<supplied pair name>"}
"""


AI_PORTFOLIO_SWITCH_PROMPT = """Output JSON immediately. Do not explain.
You are the switch-review layer for a multi-asset volatility-targeting router.
Primary task: decide whether to adjust the current strategy at this checkpoint.
If you request a switch, the selection layer will choose a new estimator-controller pair for the next ten trading days.
If you hold, the currently selected pair remains active until the next checkpoint ten trading days later.
Use the supplied volatility-regime time series and current regime forecast. A next-day regime change versus the previous day is a significant shift signal.
Use each candidate pair's 100/60/20/10-day metrics, equity curves, and rankings. Rankings prioritize Sharpe, then drawdown, then turnover, then cVaR.
Return is the first reason to switch, but only when the candidate also has acceptable Sharpe and stable drawdown behavior.
Use each pair's drawdown-consistency profile as the main risk check. Consistency means the pair's drawdowns stay stable across windows and close to the supplied target_abs_drawdown, currently about 6%.
Also inspect max_abs_drawdown and max_drawdown_target_gap separately. Favor candidates whose worst recent drawdown is controlled and close to the target.
Do not request a switch into a candidate whose recent drawdown consistency or max drawdown is materially worse than the current pair unless the current pair is clearly deteriorating and the candidate's return advantage is strong.
Use the overall routed strategy equity curve, metrics, components, active pair context, and switch log.
Decide whether the current pair and candidate set are capable of serving as the strategy for the next ten trading days.
A strong suggestion, and the most important switching guideline, is to switch only if the current pair is weak in many recent ranks, such as 60-day, 20-day, or 10-day ranks, and weak in recent metrics.
Be conservative with Sharpe and drawdown. Carefully question candidates that do not offer significant improvement.
If the current pair is still competitive across recent ranks or metrics, prefer holding unless the other evidence is clearly stronger.
Desired switch frequency is about once every 60 trading days, but request a switch when a pair can significantly improve the strategy.
Return only:
{"action": "hold"}
or:
{"action": "switch"}
"""


AI_PORTFOLIO_SENSITIVITY_SWITCH_GUIDANCE = (
    "Use the supplied sensitivity policy. High sensitivity favors switching when one candidate's "
    "short-term performance is better than the active pair and permits consecutive checkpoint switches. "
    "Medium/low sensitivity favors holding unless a group of candidates is better across multiple rank windows."
)


AI_VOLATILITY_REGIME_PROMPT = """Output JSON immediately. Do not explain.
You generate a volatility regime for a combined multi-asset portfolio.
Use only the supplied past/current time-series summaries:
multi-asset individual returns, equal-weight portfolio returns and realized volatility, VIX, term spread, and credit spreads.
Do not infer from strategy pair metrics, pair rankings, selected strategy performance, future data, or any data outside the payload.
Classify the current volatility environment as exactly one of:
risk_on, balanced, defensive.
These are volatility regimes:
risk_on means benign/low volatility, stable cross-asset returns, calm VIX, and contained credit/rate stress.
balanced means normal or mixed volatility conditions without a clear low-vol or elevated-vol signal.
defensive means elevated or rising volatility, stressed cross-asset returns, high/rising VIX, or widening credit/rate stress.
Use all three regimes when evidence supports them; keep the series diversified but data-driven.
Return only:
{"portfolio_regime": "balanced", "confidence": 0.0, "reason": "<short reason>"}
Keep reason under 24 words.
"""


class AIPortfolioRegimeRouter:
    """Route among multi-asset pair results using combined portfolio metrics."""

    responses_endpoint = "https://api.openai.com/v1/responses"
    chat_completions_endpoint = "https://api.openai.com/v1/chat/completions"
    nvidia_endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})
        self.provider = str(self.params.get("provider", "nvidia")).lower()
        self.api_format = str(self.params.get("api_format", "chat_completions")).lower()
        self.model = str(
            self.params.get("model", "")
            or os.getenv("OPENAI_MODEL", "")
            or os.getenv("CHATGPT_MODEL", "")
            or "openai/gpt-oss-120b"
        )
        api_key_env = str(self.params.get("api_key_env", "") or "")
        if self.provider == "nvidia":
            default_api_key = (
                os.getenv("NVIDIA_API_KEY", "")
                or os.getenv("NVAPI_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
                or os.getenv("OPENAI_KEY", "")
            )
        else:
            default_api_key = (
                os.getenv("OPENAI_API_KEY", "")
                or os.getenv("OPENAI_KEY", "")
                or os.getenv("NVIDIA_API_KEY", "")
                or os.getenv("NVAPI_KEY", "")
            )
        self.api_key = str(
            self.params.get("api_key", "")
            or (os.getenv(api_key_env, "") if api_key_env else "")
            or default_api_key
        )
        self.endpoint = str(self.params.get("endpoint", "") or self._default_endpoint())
        self.ai_enabled = bool(self.params.get("ai_enabled", True))
        self.timeout = float(self.params.get("timeout", 45.0))
        self.max_output_tokens = int(self.params.get("max_output_tokens", 256))
        self.ai_retries = max(int(self.params.get("ai_retries", 1)), 1)
        self.temperature = self.params.get("temperature")
        self.reasoning_effort = self.params.get("reasoning_effort")
        self.retry_backoff_base = max(float(self.params.get("retry_backoff_base", 1.5)), 0.1)
        self.rate_limit_retry_padding = max(float(self.params.get("rate_limit_retry_padding", 1.0)), 0.0)
        self.equity_snapshot_points = max(int(self.params.get("equity_snapshot_points", 12)), 2)
        self.include_equity_curves_by_window = bool(
            self.params.get("include_equity_curves_by_window", True)
        )
        self.include_strategy_context = bool(self.params.get("include_strategy_context", True))
        self.regime_interval = max(int(self.params.get("regime_interval", 20)), 1)
        self.switch_review_interval = max(int(self.params.get("switch_review_interval", 20)), 1)
        self.selection_interval = max(int(self.params.get("selection_interval", self.switch_review_interval)), 1)
        self.candidate_top_n = max(int(self.params.get("candidate_top_n", 12)), 1)
        self.regime_rank_top_n = max(int(self.params.get("regime_rank_top_n", 16)), 1)
        self.switch_cost_penalty = float(self.params.get("switch_cost_penalty", 0.001))
        self.switch_return_penalty = float(self.params.get("switch_return_penalty", self.switch_cost_penalty))
        self.apply_switch_penalty_to_returns = bool(self.params.get("apply_switch_penalty_to_returns", True))
        self.use_switch_hurdle_filter = bool(self.params.get("use_switch_hurdle_filter", True))
        self.use_regime_rank_in_decision = bool(self.params.get("use_regime_rank_in_decision", True))
        self.target_vol = float(self.params.get("target_vol", 0.10))
        self.max_candidate_vol = float(self.params.get("max_candidate_vol", 0.11))
        raw_max_recent_dd = self.params.get("max_candidate_recent_drawdown")
        self.max_candidate_recent_drawdown = (
            None if raw_max_recent_dd is None else float(raw_max_recent_dd)
        )
        self.candidate_sort_drawdown_first = bool(
            self.params.get("candidate_sort_drawdown_first", False)
        )
        self.candidate_sort_drawdown_consistency = bool(
            self.params.get("candidate_sort_drawdown_consistency", False)
        )
        raw_drawdown_target = self.params.get("drawdown_consistency_target", 0.06)
        self.drawdown_consistency_target = (
            None if raw_drawdown_target is None else float(raw_drawdown_target)
        )
        raw_max_drawdown_target = self.params.get("max_drawdown_target", self.drawdown_consistency_target)
        self.max_drawdown_target = (
            None if raw_max_drawdown_target is None else float(raw_max_drawdown_target)
        )
        self.candidate_vol_tolerance = float(self.params.get("candidate_vol_tolerance", 0.015))
        self.extreme_vol_tolerance = float(self.params.get("extreme_vol_tolerance", 0.035))
        self.return_advantage_threshold = float(self.params.get("return_advantage_threshold", 0.02))
        self.sharpe_advantage_threshold = float(self.params.get("sharpe_advantage_threshold", 0.20))
        self.regime_rank_candidate_top_k = max(int(self.params.get("regime_rank_candidate_top_k", 8)), 1)
        self.overall_rank_top_k = max(int(self.params.get("overall_rank_top_k", 8)), 1)
        hold_gate_days = self.params.get("hold_gate_days", self.params.get("cooldown_hold_days", 0))
        self.cooldown_hold_days = max(int(hold_gate_days), 0)
        self.initial_hold_days = max(int(self.params.get("initial_hold_days", 0)), 0)
        raw_cooldown_windows = self.params.get("cooldown_rank_windows", [60, 20])
        if isinstance(raw_cooldown_windows, str):
            raw_cooldown_windows = [x.strip() for x in raw_cooldown_windows.split(",") if x.strip()]
        self.cooldown_rank_windows = sorted({max(int(x), 1) for x in raw_cooldown_windows}, reverse=True)
        self.cooldown_poor_rank_threshold = max(
            int(self.params.get("cooldown_poor_rank_threshold", self.overall_rank_top_k)),
            1,
        )
        self.cooldown_missing_rank_is_poor = bool(
            self.params.get("cooldown_missing_rank_is_poor", False)
        )
        self.fallback_switch_when_all_recent_ranks_poor = bool(
            self.params.get("fallback_switch_when_all_recent_ranks_poor", False)
        )
        self.overall_rank_window = max(int(self.params.get("overall_rank_window", 60)), 1)
        raw_windows = self.params.get("recent_rank_windows", [100, 60, 20])
        if isinstance(raw_windows, str):
            raw_windows = [x.strip() for x in raw_windows.split(",") if x.strip()]
        self.recent_rank_windows = sorted({max(int(x), 1) for x in raw_windows}, reverse=True)
        raw_sensitivity = self.params.get("sensitivity")
        self.sensitivity_enabled = self._sensitivity_is_enabled(raw_sensitivity)
        self.sensitivity = (
            self._normalize_sensitivity(raw_sensitivity)
            if self.sensitivity_enabled
            else None
        )
        self.sensitivity_config = (
            self._sensitivity_config(str(self.sensitivity))
            if self.sensitivity_enabled
            else {}
        )
        self.sharpe_rank_tie_band = max(float(self.params.get("sharpe_rank_tie_band", 0.05)), 1e-6)
        self.require_recent_momentum_improving = bool(self.params.get("require_recent_momentum_improving", True))
        self._last_regime: dict[str, Any] | None = None
        self._last_regime_step = -10**9
        self._last_selection_step = -10**9
        self._last_switch_review_step = -10**9
        self._last_portfolio_regime: str | None = None
        self._regime_history: list[dict[str, Any]] = []
        self._rank_review_history: list[dict[str, Any]] = []

    def _default_endpoint(self) -> str:
        if self.provider == "nvidia":
            return self.nvidia_endpoint
        if self.api_format in ("chat", "chat_completions", "chat-completions"):
            return self.chat_completions_endpoint
        return self.responses_endpoint

    @staticmethod
    def _sensitivity_is_enabled(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        return bool(text) and text not in {"off", "none", "false", "disabled", "disable", "0"}

    @staticmethod
    def _normalize_sensitivity(value: Any) -> str:
        text = str(value or "medium").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "veryhigh": "very_high",
            "very_high": "very_high",
            "high": "high",
            "mid": "medium",
            "medium": "medium",
            "normal": "medium",
            "low": "low",
            "verylow": "very_low",
            "very_low": "very_low",
        }
        return aliases.get(text, "medium")

    @staticmethod
    def _sensitivity_config(level: str) -> dict[str, Any]:
        configs = {
            "very_high": {
                "level": "very_high",
                "min_hold_days": 0,
                "group_better_required": 0,
                "group_min_windows": 1,
                "description": "Switch aggressively when one candidate has better short-term performance than the active pair; consecutive checkpoint switches are allowed.",
            },
            "high": {
                "level": "high",
                "min_hold_days": 0,
                "group_better_required": 0,
                "group_min_windows": 1,
                "description": "Switch when a candidate's short-term performance is better than the active pair; consecutive checkpoint switches are allowed.",
            },
            "medium": {
                "level": "medium",
                "min_hold_days": 30,
                "group_better_required": 2,
                "group_min_windows": 2,
                "description": "Prefer holding for at least 30 days unless a group of candidates beats the active pair across multiple rank windows.",
            },
            "low": {
                "level": "low",
                "min_hold_days": 60,
                "group_better_required": 2,
                "group_min_windows": 2,
                "description": "Prefer holding for at least 60 days unless a group of candidates beats the active pair across multiple rank windows.",
            },
            "very_low": {
                "level": "very_low",
                "min_hold_days": 90,
                "group_better_required": 3,
                "group_min_windows": 2,
                "description": "Switch rarely; prefer holding for at least 90 days unless several candidates beat the active pair across multiple rank windows.",
            },
        }
        return configs.get(level, configs["medium"])

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match is None:
                raise ValueError(f"AI response was not JSON: {text[:200]!r}")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("AI response JSON must be an object.")
        return parsed

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", choices[0].get("text", ""))
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str) and text.strip():
                            chunks.append(text)
                    elif isinstance(item, str) and item.strip():
                        chunks.append(item)
                if chunks:
                    return "\n".join(chunks)
            for key in ("reasoning_content", "reasoning"):
                text = msg.get(key)
                if isinstance(text, str) and "{" in text and "}" in text:
                    return text
            text = choices[0].get("text")
            if isinstance(text, str) and text.strip():
                return text
        chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()

    def _call_ai(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("AI API key is empty.")
        if self.api_format in ("chat", "chat_completions", "chat-completions") or self.provider == "nvidia":
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_output_tokens,
            }
            if self.temperature is not None:
                payload["temperature"] = float(self.temperature)
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload = {
                "model": self.model,
                "instructions": system_prompt,
                "input": prompt,
                "max_output_tokens": self.max_output_tokens,
            }
            if self.temperature is not None:
                payload["temperature"] = float(self.temperature)
            if self.reasoning_effort:
                payload["reasoning"] = {"effort": self.reasoning_effort}
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(self.ai_retries):
            retry_sleep: float | None = None
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return self._parse_json(self._extract_text(data))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"AI request failed: HTTP {exc.code}: {detail}")
                retry_sleep = self._retry_sleep_seconds(exc, detail)
            except URLError as exc:
                last_error = RuntimeError(f"AI request failed: {exc.reason}")
            except Exception as exc:
                last_error = exc
            if attempt + 1 < self.ai_retries:
                if retry_sleep is None:
                    retry_sleep = self.retry_backoff_base * (2 ** attempt)
                time.sleep(retry_sleep)
        raise RuntimeError(f"AI request failed after {self.ai_retries} attempt(s): {last_error}") from last_error

    def _retry_sleep_seconds(self, exc: HTTPError, detail: str) -> float | None:
        if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504}:
            return None
        retry_after = exc.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after) + self.rate_limit_retry_padding, 0.0)
            except ValueError:
                pass
        match = re.search(r"try again in\s+([0-9.]+)\s*(ms|s)", detail, flags=re.I)
        if match:
            value = float(match.group(1))
            seconds = value / 1000.0 if match.group(2).lower() == "ms" else value
            return max(seconds + self.rate_limit_retry_padding, 0.0)
        return None

    def _fallback_regime(self, metrics: dict[str, Any]) -> dict[str, Any]:
        vol = float(metrics.get("realized_vol", 0.0))
        drawdown = float(metrics.get("drawdown", 0.0))
        vol_error = abs(float(metrics.get("vol_tracking_error", 0.0)))
        if drawdown <= -0.05 or vol_error > 0.04 or vol > 0.13:
            regime = "defensive"
        elif float(metrics.get("rolling_sharpe", 0.0)) > 0.8 and vol_error < 0.025:
            regime = "risk_on"
        else:
            regime = "balanced"
        return {"portfolio_regime": regime, "confidence": 0.0, "reason": "Deterministic combined portfolio metrics fallback"}

    def classify_regime(
        self,
        step: int,
        active_pair: str,
        active_metrics: dict[str, Any],
        all_metrics: dict[str, dict[str, Any]] | None = None,
        regime_rankings: dict[str, Any] | None = None,
        date: Any = None,
    ) -> dict[str, Any]:
        if self._last_regime is not None and step - self._last_regime_step < self.regime_interval:
            return dict(self._last_regime)
        dispersion = _pair_metric_dispersion(all_metrics or {})
        payload = {
            "instruction": "Classify the current combined portfolio regime. Do not choose a pair.",
            "date": str(date)[:10] if date is not None else "",
            "active_pair": active_pair,
            "active_portfolio_metrics": active_metrics,
            "pair_metric_dispersion": dispersion,
            "recent_ai_regime_history": self._regime_history[-8:],
            "regime_rank_blocks": _compact_regime_rank_blocks(regime_rankings, top_n=6),
        }
        try:
            if not self.ai_enabled:
                raise ValueError("AI disabled")
            regime = self._call_ai(json.dumps(payload, default=str), AI_PORTFOLIO_REGIME_PROMPT)
            if regime.get("portfolio_regime") not in set(PORTFOLIO_REGIMES):
                raise ValueError("Invalid portfolio regime")
        except Exception as exc:
            regime = self._fallback_regime(active_metrics)
            regime["error"] = str(exc)
        self._last_regime = dict(regime)
        self._last_regime_step = step
        self._regime_history.append(
            {
                "step": step,
                "date": str(date)[:10] if date is not None else "",
                "portfolio_regime": regime.get("portfolio_regime"),
                "confidence": regime.get("confidence", 0.0),
                "reason": regime.get("reason", ""),
            }
        )
        return regime

    def _score_metrics(self, metrics: dict[str, Any], regime: str) -> float:
        vol_error = abs(float(metrics.get("vol_tracking_error", 0.0)))
        drawdown = abs(min(float(metrics.get("drawdown", 0.0)), 0.0))
        turnover = float(metrics.get("turnover", 0.0))
        sharpe = float(metrics.get("rolling_sharpe", 0.0))
        ret = float(metrics.get("trailing_return", 0.0))
        ann_ret = float(metrics.get("annualized_return", ret * 4.0))
        drawdown_weight = 0.35 if regime in {"risk_on", "balanced"} else 0.70
        vol_penalty = max(float(metrics.get("realized_vol", 0.0)) - self.max_candidate_vol, 0.0)
        return drawdown_weight * drawdown + 1.4 * vol_penalty + 0.05 * turnover - 0.75 * sharpe - 1.75 * ann_ret - 0.08 * ret

    def _rank_candidates(self, all_metrics: dict[str, dict[str, Any]], regime: str) -> list[dict[str, Any]]:
        rows = []
        for pair, m in all_metrics.items():
            row = dict(m)
            row.update({"pair": pair, "score": self._score_metrics(m, regime)})
            rows.append(row)
        rows.sort(key=lambda x: x["score"])
        return rows[: self.candidate_top_n]

    def observe_rankings(self, regime_rankings: dict[str, Any] | None, step: int, date: Any = None) -> None:
        snapshot: dict[str, Any] = {"step": step, "date": str(date)[:10] if date is not None else "", "ranks": {}}
        for regime in PORTFOLIO_REGIMES:
            snapshot["ranks"][regime] = {
                str(row.get("pair")): int(row.get("selection_rank", i + 1))
                for i, row in enumerate(_rank_rows_for_regime(regime_rankings, regime))
            }
        self._rank_review_history.append(snapshot)
        self._rank_review_history = self._rank_review_history[-8:]

    def _rank_consistency(self, pair: str, regime: str, top_k: int = 8) -> dict[str, Any]:
        ranks = []
        for snapshot in self._rank_review_history[-6:]:
            rank = snapshot.get("ranks", {}).get(regime, {}).get(pair)
            if rank is not None:
                ranks.append(int(rank))
        if not ranks:
            return {"rank_consistency_count": 0, "rank_consistency_score": 0.0, "avg_recent_rank": None}
        top_hits = sum(rank <= top_k for rank in ranks)
        score = top_hits / len(ranks)
        return {
            "rank_consistency_count": int(top_hits),
            "rank_consistency_score": float(score),
            "avg_recent_rank": float(np.mean(ranks)),
        }

    def _candidate_pool(
        self,
        active_pair: str,
        all_metrics: dict[str, dict[str, Any]],
        regime: str,
        regime_rankings: dict[str, Any] | None = None,
        overall_rankings: list[dict[str, Any]] | None = None,
        recent_rankings: dict[int, list[dict[str, Any]]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        regime_rows = _rank_rows_for_regime(regime_rankings, regime)
        regime_rank_by_pair = {str(row.get("pair")): int(row.get("selection_rank", i + 1)) for i, row in enumerate(regime_rows)}
        regime_stats_by_pair = {str(row.get("pair")): dict(row) for row in regime_rows}
        overall_rows = list(overall_rankings or [])
        overall_rank_by_pair = {str(row.get("pair")): int(row.get("overall_rank", row.get("selection_rank", i + 1))) for i, row in enumerate(overall_rows)}
        overall_stats_by_pair = {str(row.get("pair")): dict(row) for row in overall_rows}
        recent_rankings = recent_rankings or {}
        recent_rank_by_window = {
            int(window): {str(row.get("pair")): int(row.get("overall_rank", row.get("selection_rank", i + 1))) for i, row in enumerate(rows)}
            for window, rows in recent_rankings.items()
        }
        allowed_rank_pairs: set[str] = set()
        if self.use_regime_rank_in_decision:
            allowed_rank_pairs.update(
                str(row.get("pair"))
                for row in regime_rows[: self.regime_rank_candidate_top_k]
                if str(row.get("pair")) in all_metrics
            )
        allowed_rank_pairs.update(
            str(row.get("pair"))
            for row in overall_rows[: self.overall_rank_top_k]
            if str(row.get("pair")) in all_metrics
        )
        for rows in recent_rankings.values():
            allowed_rank_pairs.update(
                str(row.get("pair"))
                for row in rows[: self.overall_rank_top_k]
                if str(row.get("pair")) in all_metrics
            )
        candidates = self._rank_candidates(all_metrics, regime)
        if allowed_rank_pairs:
            candidates = [row for row in candidates if str(row.get("pair")) in allowed_rank_pairs]
        for row in candidates:
            pair = str(row.get("pair"))
            if self.use_regime_rank_in_decision and pair in regime_stats_by_pair:
                row["regime_selection_rank"] = regime_rank_by_pair.get(pair)
                row["regime_rank_statistics"] = regime_stats_by_pair[pair]
            if pair in overall_stats_by_pair:
                row["overall_recent_rank"] = overall_rank_by_pair.get(pair)
                row["overall_recent_rank_statistics"] = overall_stats_by_pair[pair]
            for window, rank_by_pair in recent_rank_by_window.items():
                if pair in rank_by_pair:
                    row[f"overall_recent_{window}d_rank"] = rank_by_pair[pair]
            row.update(self._rank_consistency(pair, regime))
        regime_candidates = [row for row in regime_rows if str(row.get("pair")) in all_metrics]
        if self.use_regime_rank_in_decision:
            for rank_row in regime_candidates[: self.regime_rank_candidate_top_k]:
                pair = str(rank_row.get("pair"))
                if all(str(c.get("pair")) != pair for c in candidates):
                    merged = dict(all_metrics[pair])
                    merged["pair"] = pair
                    merged["score"] = self._score_metrics(merged, regime)
                    merged["regime_selection_rank"] = regime_rank_by_pair.get(pair)
                    merged["regime_rank_statistics"] = dict(rank_row)
                    if pair in overall_stats_by_pair:
                        merged["overall_recent_rank"] = overall_rank_by_pair.get(pair)
                        merged["overall_recent_rank_statistics"] = overall_stats_by_pair[pair]
                    for window, rank_by_pair in recent_rank_by_window.items():
                        if pair in rank_by_pair:
                            merged[f"overall_recent_{window}d_rank"] = rank_by_pair[pair]
                    merged.update(self._rank_consistency(pair, regime))
                    candidates.append(merged)
        for rank_row in overall_rows[: self.overall_rank_top_k]:
            pair = str(rank_row.get("pair"))
            if pair in all_metrics and all(str(c.get("pair")) != pair for c in candidates):
                merged = dict(all_metrics[pair])
                merged["pair"] = pair
                merged["score"] = self._score_metrics(merged, regime)
                merged["regime_selection_rank"] = regime_rank_by_pair.get(pair)
                if pair in regime_stats_by_pair:
                    merged["regime_rank_statistics"] = regime_stats_by_pair[pair]
                merged["overall_recent_rank"] = overall_rank_by_pair.get(pair)
                merged["overall_recent_rank_statistics"] = dict(rank_row)
                for window, rank_by_pair in recent_rank_by_window.items():
                    if pair in rank_by_pair:
                        merged[f"overall_recent_{window}d_rank"] = rank_by_pair[pair]
                merged.update(self._rank_consistency(pair, regime))
                candidates.append(merged)
        for window, rows in recent_rankings.items():
            for rank_row in rows[: self.overall_rank_top_k]:
                pair = str(rank_row.get("pair"))
                if pair in all_metrics and all(str(c.get("pair")) != pair for c in candidates):
                    merged = dict(all_metrics[pair])
                    merged["pair"] = pair
                    merged["score"] = self._score_metrics(merged, regime)
                    merged["overall_recent_rank"] = overall_rank_by_pair.get(pair)
                    merged[f"overall_recent_{int(window)}d_rank"] = recent_rank_by_window.get(int(window), {}).get(pair)
                    merged[f"overall_recent_{int(window)}d_rank_statistics"] = dict(rank_row)
                    merged.update(self._rank_consistency(pair, regime))
                    candidates.append(merged)
        if self.require_recent_momentum_improving:
            candidates = [row for row in candidates if bool(row.get("recent_momentum_improving", False))]
        if self.max_candidate_recent_drawdown is not None:
            candidates = [
                row for row in candidates
                if float(row.get("drawdown", -np.inf)) >= self.max_candidate_recent_drawdown
            ]

        def vol_bucket(row: dict[str, Any]) -> tuple[int, float]:
            vol = float(row.get("realized_vol", self.target_vol))
            over_cap = max(vol - self.max_candidate_vol, 0.0)
            return (0 if over_cap <= 1e-12 else 1, over_cap)

        if self.candidate_sort_drawdown_first:
            sort_key = lambda row: (
                vol_bucket(row),
                -float(row.get("drawdown", -np.inf)),
                -float(row.get("rolling_sharpe", 0.0)),
                -float(row.get("annualized_return", 0.0)),
                -float(row.get("rank_consistency_score", 0.0)),
                regime_rank_by_pair.get(str(row.get("pair")), 10**9),
                overall_rank_by_pair.get(str(row.get("pair")), 10**9),
                float(row.get("score", 0.0)),
            )
        elif self.candidate_sort_drawdown_consistency:
            sort_key = lambda row: (
                vol_bucket(row),
                -float(row.get("annualized_return", 0.0)),
                -float(row.get("rolling_sharpe", 0.0)),
                float(row.get("max_drawdown_excess_over_target", np.inf)),
                float(row.get("max_drawdown_target_gap", np.inf)),
                -float(row.get("drawdown_consistency_score", -np.inf)),
                regime_rank_by_pair.get(str(row.get("pair")), 10**9),
                overall_rank_by_pair.get(str(row.get("pair")), 10**9),
                float(row.get("score", 0.0)),
            )
        else:
            sort_key = lambda row: (
                vol_bucket(row),
                -float(row.get("annualized_return", 0.0)),
                -float(row.get("rolling_sharpe", 0.0)),
                -float(row.get("rank_consistency_score", 0.0)),
                regime_rank_by_pair.get(str(row.get("pair")), 10**9),
                overall_rank_by_pair.get(str(row.get("pair")), 10**9),
                float(row.get("score", 0.0)),
            )
        candidates = sorted(candidates, key=sort_key)[: self.candidate_top_n]
        return candidates, regime_rank_by_pair

    def _candidate_clears_switch_hurdle(self, candidate: dict[str, Any], active_metrics: dict[str, Any]) -> bool:
        if float(candidate.get("realized_vol", np.inf)) > self.max_candidate_vol:
            return False
        ret_adv = float(candidate.get("annualized_return", 0.0)) - float(active_metrics.get("annualized_return", 0.0))
        sharpe_adv = float(candidate.get("rolling_sharpe", 0.0)) - float(active_metrics.get("rolling_sharpe", 0.0))
        return ret_adv >= self.return_advantage_threshold + self.switch_return_penalty or sharpe_adv >= self.sharpe_advantage_threshold

    def _sensitivity_group_better(
        self,
        active_pair: str,
        candidate_pairs: list[dict[str, Any]],
        recent_rankings: dict[int, list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        rank_by_window: dict[int, dict[str, int]] = {}
        for window, rows in (recent_rankings or {}).items():
            rank_by_window[int(window)] = {
                str(row.get("pair")): int(row.get("overall_rank", row.get("selection_rank", i + 1)))
                for i, row in enumerate(rows)
            }
        min_windows = int(self.sensitivity_config.get("group_min_windows", 2))
        required = int(self.sensitivity_config.get("group_better_required", 0))
        better_rows = []
        for candidate in candidate_pairs:
            pair = str(candidate.get("pair"))
            better_windows = []
            for window, ranks in rank_by_window.items():
                candidate_rank = ranks.get(pair)
                active_rank = ranks.get(active_pair)
                if candidate_rank is not None and (active_rank is None or candidate_rank < active_rank):
                    better_windows.append(
                        {
                            "window": int(window),
                            "candidate_rank": int(candidate_rank),
                            "active_rank": None if active_rank is None else int(active_rank),
                        }
                    )
            if len(better_windows) >= min_windows:
                better_rows.append({"pair": pair, "better_windows": better_windows})
        return {
            "required_group_size": required,
            "min_windows_per_pair": min_windows,
            "better_group_size": int(len(better_rows)),
            "gate_passed": bool(required <= 0 or len(better_rows) >= required),
            "better_pairs": better_rows[: max(required, 3)],
        }

    @staticmethod
    def _recent_rank_for_pair(
        pair: str,
        recent_rankings: dict[int, list[dict[str, Any]]] | None,
        window: int,
    ) -> int | None:
        for row in (recent_rankings or {}).get(int(window), []):
            if str(row.get("pair")) == pair:
                return int(row.get("overall_rank", row.get("selection_rank", 10**9)))
        return None

    def switch_review(
        self,
        step: int,
        active_pair: str,
        active_metrics: dict[str, Any],
        all_metrics: dict[str, dict[str, Any]],
        portfolio_regime: dict[str, Any],
        regime_rankings: dict[str, Any] | None = None,
        overall_rankings: list[dict[str, Any]] | None = None,
        recent_rankings: dict[int, list[dict[str, Any]]] | None = None,
        switch_log: list[dict[str, Any]] | None = None,
        strategy_context: dict[str, Any] | None = None,
        active_hold_days: int | None = None,
        pair_summaries: dict[str, dict[str, Any]] | None = None,
        pair_window_summaries: dict[str, dict[str, dict[str, Any]]] | None = None,
        pair_drawdown_profiles: dict[str, dict[str, Any]] | None = None,
        equity_curves: dict[str, list[dict[str, Any]]] | None = None,
        equity_curves_by_window: dict[int, dict[str, list[dict[str, Any]]]] | None = None,
        date: Any = None,
    ) -> dict[str, Any]:
        regime = str(portfolio_regime.get("portfolio_regime", "balanced"))
        previous_regime = self._last_portfolio_regime
        regime_changed = previous_regime is not None and previous_regime != regime
        self._last_portfolio_regime = regime
        gap = step - self._last_switch_review_step
        if gap < self.switch_review_interval:
            return {
                "action": "hold",
                "reason": f"Switch review interval hold {gap}/{self.switch_review_interval}",
                "confidence": 0.0,
                "regime_changed": regime_changed,
            }
        recent_rankings = recent_rankings or {}
        active_hold_days = int(active_hold_days or 0)
        cooldown_ranks = {
            int(window): self._recent_rank_for_pair(active_pair, recent_rankings, int(window))
            for window in self.cooldown_rank_windows
        }
        cooldown_poor_flags = {
            int(window): (
                (rank is None and self.cooldown_missing_rank_is_poor)
                or (rank is not None and rank > self.cooldown_poor_rank_threshold)
            )
            for window, rank in cooldown_ranks.items()
        }
        cooldown_all_poor = bool(cooldown_poor_flags) and all(cooldown_poor_flags.values())
        if self.cooldown_hold_days and active_hold_days < self.cooldown_hold_days and not cooldown_all_poor:
            self._last_switch_review_step = step
            return {
                "action": "hold",
                "reason": (
                    f"Cooldown hold {active_hold_days}/{self.cooldown_hold_days}; "
                    f"active ranks not poor in all windows"
                ),
                "confidence": 0.0,
                "regime_changed": regime_changed,
                "previous_portfolio_regime": previous_regime,
                "cooldown_active": True,
                "hold_gate_active": True,
                "active_hold_days": active_hold_days,
                "cooldown_ranks": cooldown_ranks,
                "cooldown_poor_flags": cooldown_poor_flags,
                "cooldown_all_poor": cooldown_all_poor,
            }

        candidates, regime_rank_by_pair = self._candidate_pool(active_pair, all_metrics, regime, regime_rankings, overall_rankings, recent_rankings)
        viable_candidates = [
            row for row in candidates
            if str(row.get("pair")) != active_pair
            and (not self.use_switch_hurdle_filter or self._candidate_clears_switch_hurdle(row, active_metrics))
        ]
        if not viable_candidates:
            self._last_switch_review_step = step
            return {
                "action": "hold",
                "reason": "No candidate available",
                "confidence": 0.0,
                "regime_changed": regime_changed,
                "previous_portfolio_regime": previous_regime,
            }
        sensitivity_group_check: dict[str, Any] = {}
        sensitivity_min_hold_days = 0
        if self.sensitivity_enabled:
            sensitivity_group_check = self._sensitivity_group_better(active_pair, viable_candidates, recent_rankings)
            sensitivity_min_hold_days = int(self.sensitivity_config.get("min_hold_days", 0))
        if (
            self.sensitivity_enabled
            and sensitivity_min_hold_days
            and active_hold_days < sensitivity_min_hold_days
            and not sensitivity_group_check.get("gate_passed", False)
        ):
            self._last_switch_review_step = step
            return {
                "action": "hold",
                "reason": (
                    f"Sensitivity {self.sensitivity} hold {active_hold_days}/{sensitivity_min_hold_days}; "
                    "group-better rank exception not met"
                ),
                "confidence": 0.0,
                "regime_changed": regime_changed,
                "previous_portfolio_regime": previous_regime,
                "sensitivity": self.sensitivity,
                "sensitivity_config": self.sensitivity_config,
                "sensitivity_group_check": sensitivity_group_check,
                "sensitivity_hold_active": True,
                "active_hold_days": active_hold_days,
            }
        active_rank = regime_rank_by_pair.get(active_pair)
        payload = {
            "instruction": "Decide whether to hold active pair or request switch review. Do not choose the new pair.",
            "date": str(date)[:10] if date is not None else "",
            "current_portfolio_regime": portfolio_regime.get("portfolio_regime", "balanced"),
            "previous_portfolio_regime": previous_regime,
            "regime_changed": regime_changed,
            "rank_rule": f"Recent ranks: Sharpe first; if Sharpe is within {self.sharpe_rank_tie_band:.3f}, prefer better drawdown, then lower turnover, then better cVaR.",
            "active_pair": active_pair,
            "active_hold_days": active_hold_days,
            "cooldown_gate": {
                "description": "Hold gate controlling switch frequency. It blocks switching before hold_days unless active pair ranks are poor in every configured rank window.",
                "hold_days": self.cooldown_hold_days,
                "rank_windows": self.cooldown_rank_windows,
                "poor_rank_threshold": self.cooldown_poor_rank_threshold,
                "active_ranks": cooldown_ranks,
                "active_rank_is_poor_by_window": cooldown_poor_flags,
                "active_rank_poor_in_all_windows": cooldown_all_poor,
            },
            "active_metrics": active_metrics,
            "active_pair_summary": (pair_summaries or {}).get(active_pair, {}),
            "active_pair_window_metrics": (pair_window_summaries or {}).get(active_pair, {}),
            "active_drawdown_consistency": (pair_drawdown_profiles or {}).get(active_pair, {}),
            "active_equity_curve": (equity_curves or {}).get(active_pair, []),
            "top_candidates_for_comparison": viable_candidates[: self.candidate_top_n],
            "candidate_window_metrics": {
                str(row.get("pair")): (pair_window_summaries or {}).get(str(row.get("pair")), {})
                for row in viable_candidates[: self.candidate_top_n]
            },
            "candidate_drawdown_consistency": {
                str(row.get("pair")): (pair_drawdown_profiles or {}).get(str(row.get("pair")), {})
                for row in viable_candidates[: self.candidate_top_n]
            },
            "candidate_equity_curves": {
                str(row.get("pair")): (equity_curves or {}).get(str(row.get("pair")), [])
                for row in viable_candidates[: self.candidate_top_n]
            },
            "recent_overall_ranks": {
                f"{int(window)}d": list(rows)[: self.overall_rank_top_k]
                for window, rows in recent_rankings.items()
            },
            "current_regime_pair_rankings": _rank_rows_for_regime(regime_rankings, regime)[: self.regime_rank_top_n],
            "regime_rank_blocks": _compact_regime_rank_blocks(regime_rankings, top_n=self.regime_rank_top_n),
            "recent_ai_regime_history": self._regime_history[-8:],
            "switch_log": list(switch_log or [])[-12:],
        }
        switch_prompt = AI_PORTFOLIO_SWITCH_PROMPT
        if self.sensitivity_enabled:
            payload.update(
                {
                    "sensitivity": self.sensitivity,
                    "sensitivity_policy": self.sensitivity_config,
                    "sensitivity_group_check": sensitivity_group_check,
                }
            )
            switch_prompt = (
                AI_PORTFOLIO_SWITCH_PROMPT
                + "\n"
                + AI_PORTFOLIO_SENSITIVITY_SWITCH_GUIDANCE
                + "\n"
            )
        if self.include_equity_curves_by_window:
            payload["active_equity_curves_by_window"] = {
                f"{int(window)}d": curves.get(active_pair, [])
                for window, curves in (equity_curves_by_window or {}).items()
            }
            payload["candidate_equity_curves_by_window"] = {
                f"{int(window)}d": {
                    str(row.get("pair")): curves.get(str(row.get("pair")), [])
                    for row in viable_candidates[: self.candidate_top_n]
                }
                for window, curves in (equity_curves_by_window or {}).items()
            }
        if self.include_strategy_context:
            payload["routed_strategy_context_through_previous_day"] = strategy_context or {}
        try:
            if not self.ai_enabled:
                raise ValueError("AI disabled")
            decision = self._call_ai(json.dumps(payload, default=str), switch_prompt)
            if decision.get("action") not in {"hold", "switch"}:
                raise ValueError("Switch review action must be hold or switch")
        except Exception as exc:
            best = viable_candidates[0] if viable_candidates else {}
            fallback_pair = str(best.get("pair", "")) if best else ""
            should_fallback_switch = bool(
                self.fallback_switch_when_all_recent_ranks_poor
                and cooldown_all_poor
                and fallback_pair
                and fallback_pair != active_pair
            )
            decision = {
                "action": "switch" if should_fallback_switch else "hold",
                "pair": fallback_pair if should_fallback_switch else active_pair,
                "reason": (
                    "AI unavailable; active pair poor in all recent ranks, switching to top viable candidate"
                    if should_fallback_switch
                    else "AI unavailable or invalid response; hold active pair"
                ),
                "confidence": 0.0,
                "error": str(exc),
            }
        self._last_switch_review_step = step
        decision["regime_changed"] = regime_changed
        decision["previous_portfolio_regime"] = previous_regime
        decision["cooldown_active"] = False
        decision["hold_gate_active"] = False
        decision["active_hold_days"] = active_hold_days
        decision["cooldown_ranks"] = cooldown_ranks
        decision["cooldown_poor_flags"] = cooldown_poor_flags
        decision["cooldown_all_poor"] = cooldown_all_poor
        return decision

    def choose_pair(
        self,
        step: int,
        active_pair: str,
        active_metrics: dict[str, Any],
        all_metrics: dict[str, dict[str, Any]],
        portfolio_regime: dict[str, Any],
        regime_rankings: dict[str, Any] | None = None,
        overall_rankings: list[dict[str, Any]] | None = None,
        recent_rankings: dict[int, list[dict[str, Any]]] | None = None,
        switch_log: list[dict[str, Any]] | None = None,
        pair_summaries: dict[str, dict[str, Any]] | None = None,
        pair_window_summaries: dict[str, dict[str, dict[str, Any]]] | None = None,
        pair_drawdown_profiles: dict[str, dict[str, Any]] | None = None,
        equity_curves: dict[str, list[dict[str, Any]]] | None = None,
        equity_curves_by_window: dict[int, dict[str, list[dict[str, Any]]]] | None = None,
        date: Any = None,
    ) -> dict[str, Any]:
        regime = str(portfolio_regime.get("portfolio_regime", "balanced"))
        candidates, regime_rank_by_pair = self._candidate_pool(active_pair, all_metrics, regime, regime_rankings, overall_rankings, recent_rankings)
        if not candidates:
            return {"action": "hold", "pair": active_pair, "reason": "No candidates", "confidence": 0.0}
        candidate_pairs = [
            row for row in candidates
            if str(row.get("pair")) != active_pair
            and (not self.use_switch_hurdle_filter or self._candidate_clears_switch_hurdle(row, active_metrics))
        ][: self.candidate_top_n]
        if not candidate_pairs:
            return {"action": "hold", "pair": active_pair, "reason": "No candidates clear switch hurdle", "confidence": 0.0}
        supplied_pairs = {str(row.get("pair")) for row in candidate_pairs}
        recent_rankings = recent_rankings or {}
        payload = {
            "instruction": "Choose which candidate to switch to. Compare candidates with current active pair.",
            "date": str(date)[:10] if date is not None else "",
            "current_portfolio_regime": portfolio_regime.get("portfolio_regime", "balanced"),
            "previous_portfolio_regime": self._last_portfolio_regime,
            "rank_rule": f"Recent ranks: Sharpe first; if Sharpe is within {self.sharpe_rank_tie_band:.3f}, prefer better drawdown, then lower turnover, then better cVaR.",
            "active_pair": active_pair,
            "active_metrics": active_metrics,
            "active_pair_summary": (pair_summaries or {}).get(active_pair, {}),
            "active_pair_window_metrics": (pair_window_summaries or {}).get(active_pair, {}),
            "active_drawdown_consistency": (pair_drawdown_profiles or {}).get(active_pair, {}),
            "active_equity_curve": (equity_curves or {}).get(active_pair, []),
            "candidate_pairs": candidate_pairs,
            "recent_overall_ranks": {
                f"{int(window)}d": list(rows)[: self.overall_rank_top_k]
                for window, rows in recent_rankings.items()
            },
            "pair_summaries_for_candidates": {
                str(row.get("pair")): (pair_summaries or {}).get(str(row.get("pair")), {})
                for row in candidate_pairs
            },
            "window_metrics_for_candidates": {
                str(row.get("pair")): (pair_window_summaries or {}).get(str(row.get("pair")), {})
                for row in candidate_pairs
            },
            "drawdown_consistency_for_candidates": {
                str(row.get("pair")): (pair_drawdown_profiles or {}).get(str(row.get("pair")), {})
                for row in candidate_pairs
            },
            "candidate_equity_curves": {
                str(row.get("pair")): (equity_curves or {}).get(str(row.get("pair")), [])
                for row in candidate_pairs
            },
            "current_regime_pair_rankings": _rank_rows_for_regime(regime_rankings, regime)[: self.regime_rank_top_n],
            "regime_rank_blocks": _compact_regime_rank_blocks(regime_rankings, top_n=self.regime_rank_top_n),
            "recent_ai_regime_history": self._regime_history[-8:],
            "switch_log": list(switch_log or [])[-12:],
        }
        if self.include_equity_curves_by_window:
            payload["active_equity_curves_by_window"] = {
                f"{int(window)}d": curves.get(active_pair, [])
                for window, curves in (equity_curves_by_window or {}).items()
            }
            payload["candidate_equity_curves_by_window"] = {
                f"{int(window)}d": {
                    str(row.get("pair")): curves.get(str(row.get("pair")), [])
                    for row in candidate_pairs
                }
                for window, curves in (equity_curves_by_window or {}).items()
            }
        try:
            if not self.ai_enabled:
                raise ValueError("AI disabled")
            decision = self._call_ai(json.dumps(payload, default=str), AI_PORTFOLIO_SELECTION_PROMPT)
            if decision.get("pair") not in supplied_pairs:
                raise ValueError("AI selected pair outside candidate set")
            decision["action"] = "switch"
        except Exception as exc:
            decision = {
                "action": "hold",
                "pair": active_pair,
                "reason": "AI selection unavailable or invalid response; hold active pair",
                "confidence": 0.0,
                "error": str(exc),
            }
        self._last_selection_step = step
        return decision


def rolling_metrics(frame: pd.DataFrame, end: int, window: int = 63) -> dict[str, Any]:
    hist = frame.iloc[max(0, end - window + 1) : end + 1]
    r = hist["returns_with_rf"].fillna(0.0)
    r20 = frame.iloc[max(0, end - 20 + 1) : end + 1]["returns_with_rf"].fillna(0.0)
    r40 = frame.iloc[max(0, end - 40 + 1) : end + 1]["returns_with_rf"].fillna(0.0)
    eq = np.exp(r.cumsum())
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    drawdown = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
    target = float(hist["target_vol"].dropna().iloc[-1]) if "target_vol" in hist and hist["target_vol"].notna().any() else 0.10
    realized = float(ann_vol)
    ret20 = float(r20.sum()) if len(r20) else 0.0
    ret40 = float(r40.sum()) if len(r40) else 0.0
    avg20 = ret20 / max(len(r20), 1)
    avg40 = ret40 / max(len(r40), 1)
    return {
        "obs": int(len(hist)),
        "trailing_return": float(r.sum()),
        "recent_return_20d": ret20,
        "recent_return_40d": ret40,
        "recent_momentum_improving": bool(len(r20) >= 10 and len(r40) >= 20 and ret20 > 0.0 and avg20 >= avg40),
        "annualized_return": ann_ret,
        "rolling_sharpe": sharpe,
        "drawdown": drawdown,
        "cvar_5": _cvar(r),
        "realized_vol": realized,
        "vol_tracking_error": realized - target,
        "turnover": float(hist.get("turnover", pd.Series(0.0, index=hist.index)).mean()),
    }


def _regime_from_metrics(metrics: dict[str, Any]) -> str:
    vol = float(metrics.get("realized_vol", 0.0))
    drawdown = float(metrics.get("drawdown", 0.0))
    vol_error = abs(float(metrics.get("vol_tracking_error", 0.0)))
    sharpe = float(metrics.get("rolling_sharpe", 0.0))
    ann_ret = float(metrics.get("annualized_return", 0.0))
    if sharpe >= 0.85 and ann_ret >= 0.08 and 0.075 <= vol <= 0.115:
        return "risk_on"
    if drawdown <= -0.07 or vol_error >= 0.035 or ann_ret < 0.02:
        return "defensive"
    if sharpe >= 0.65 and ann_ret >= 0.05 and vol_error <= 0.03:
        return "risk_on"
    return "balanced"


def _pair_metric_dispersion(all_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not all_metrics:
        return {}
    out: dict[str, Any] = {"pair_count": len(all_metrics)}
    for key in ("rolling_sharpe", "drawdown", "realized_vol", "vol_tracking_error", "turnover", "trailing_return"):
        values = np.asarray([float(m.get(key, 0.0)) for m in all_metrics.values()], dtype=float)
        if len(values) == 0:
            continue
        out[f"{key}_median"] = float(np.nanmedian(values))
        out[f"{key}_p25"] = float(np.nanpercentile(values, 25))
        out[f"{key}_p75"] = float(np.nanpercentile(values, 75))
    return out


def _rank_rows_for_regime(regime_rankings: dict[str, Any] | None, regime: str) -> list[dict[str, Any]]:
    block = (regime_rankings or {}).get(regime, [])
    if isinstance(block, dict):
        rows = block.get("pairs", [])
    else:
        rows = block
    return list(rows) if isinstance(rows, list) else []


def _compact_regime_rank_blocks(regime_rankings: dict[str, Any] | None, top_n: int = 8) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for regime in PORTFOLIO_REGIMES:
        block = (regime_rankings or {}).get(regime, {})
        if isinstance(block, dict):
            copied = {k: v for k, v in block.items() if k != "pairs"}
            copied["pairs"] = _rank_rows_for_regime(regime_rankings, regime)[:top_n]
            out[regime] = copied
        else:
            out[regime] = {
                "regime": regime,
                "rank_rule": "Sharpe first; within Sharpe tie band, drawdown, turnover, cVaR",
                "pairs": _rank_rows_for_regime(regime_rankings, regime)[:top_n],
            }
    return out


def _cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return 0.0
    cutoff = float(r.quantile(alpha))
    tail = r[r <= cutoff]
    return float(tail.mean()) if len(tail) else cutoff


def _sharpe_tie_rank_key(row: dict[str, Any], tie_band: float = 0.05) -> tuple[float, float, float, float, float]:
    sharpe = float(row.get("rolling_sharpe", 0.0))
    drawdown = abs(min(float(row.get("drawdown", 0.0)), 0.0))
    turnover = float(row.get("turnover", 0.0))
    cvar = abs(min(float(row.get("cvar_5", 0.0)), 0.0))
    ann_ret = float(row.get("annualized_return", 0.0))
    sharpe_band = -np.floor(sharpe / max(tie_band, 1e-6))
    return (float(sharpe_band), drawdown, turnover, cvar, -ann_ret)


def pair_summary_metrics(frame: pd.DataFrame, end: int) -> dict[str, Any]:
    hist = frame.iloc[: end + 1]
    if hist.empty:
        return {}
    r = hist["returns_with_rf"].fillna(0.0)
    eq = np.exp(r.cumsum())
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    drawdown = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
    target = float(hist["target_vol"].dropna().iloc[-1]) if "target_vol" in hist and hist["target_vol"].notna().any() else 0.10
    return {
        "obs": int(len(hist)),
        "trailing_return": float(r.sum()),
        "annualized_return": ann_ret,
        "realized_vol": ann_vol,
        "vol_tracking_error": ann_vol - target,
        "rolling_sharpe": sharpe,
        "drawdown": drawdown,
        "cvar_5": _cvar(r),
        "turnover": float(hist.get("turnover", pd.Series(0.0, index=hist.index)).mean()),
    }


def pair_window_metrics(frame: pd.DataFrame, end: int, window: int) -> dict[str, Any]:
    hist = frame.iloc[max(0, end - window + 1) : end + 1]
    if hist.empty:
        return {}
    r = hist["returns_with_rf"].fillna(0.0)
    eq = np.exp(r.cumsum())
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    drawdown = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
    return {
        "obs": int(len(hist)),
        "window_days": int(window),
        "trailing_return": float(r.sum()),
        "annualized_return": ann_ret,
        "realized_vol": ann_vol,
        "rolling_sharpe": sharpe,
        "drawdown": drawdown,
        "cvar_5": _cvar(r),
        "turnover": float(hist.get("turnover", pd.Series(0.0, index=hist.index)).mean()),
    }


def drawdown_consistency_profile(
    window_metrics: dict[str, dict[str, Any]],
    target_abs_drawdown: float | None = 0.07,
    max_abs_drawdown_target: float | None = None,
) -> dict[str, Any]:
    rows = []
    for label, metrics in (window_metrics or {}).items():
        if not metrics:
            continue
        drawdown = float(metrics.get("drawdown", 0.0))
        abs_drawdown = abs(min(drawdown, 0.0))
        row = {
            "window": str(label),
            "drawdown": drawdown,
            "abs_drawdown": abs_drawdown,
            "rolling_sharpe": float(metrics.get("rolling_sharpe", 0.0)),
            "annualized_return": float(metrics.get("annualized_return", 0.0)),
        }
        if target_abs_drawdown is not None:
            row["gap_to_target_abs_drawdown"] = abs(abs_drawdown - float(target_abs_drawdown))
        rows.append(row)
    if not rows:
        out = {"available": False}
        if target_abs_drawdown is not None:
            out["target_abs_drawdown"] = float(target_abs_drawdown)
        return out
    abs_values = np.array([row["abs_drawdown"] for row in rows], dtype=float)
    mean_abs = float(abs_values.mean())
    std_abs = float(abs_values.std(ddof=0)) if len(abs_values) else 0.0
    max_abs = float(abs_values.max())
    out = {
        "available": True,
        "mean_abs_drawdown": mean_abs,
        "std_abs_drawdown": std_abs,
        "min_abs_drawdown": float(abs_values.min()),
        "max_abs_drawdown": max_abs,
        "consistency_score": float(-(mean_abs + std_abs)),
        "interpretation": "Higher consistency_score is better; it favors lower and more stable absolute drawdown across recent windows.",
        "windows": rows,
    }
    if target_abs_drawdown is not None:
        gaps = np.abs(abs_values - float(target_abs_drawdown))
        out["target_abs_drawdown"] = float(target_abs_drawdown)
        out["mean_gap_to_target"] = float(gaps.mean())
        out["consistency_score"] = float(-(gaps.mean() + std_abs))
        out["interpretation"] = "Higher consistency_score is better; it favors stable drawdown near target_abs_drawdown."
    if max_abs_drawdown_target is not None:
        out["max_abs_drawdown_target"] = float(max_abs_drawdown_target)
        out["max_drawdown_target_gap"] = abs(max_abs - float(max_abs_drawdown_target))
        out["max_drawdown_excess_over_target"] = max(max_abs - float(max_abs_drawdown_target), 0.0)
    return out


def routed_strategy_context(
    records: list[dict[str, Any]],
    returns: list[float],
    as_of_date: Any,
    windows: list[int] | tuple[int, ...] = (100, 60, 20),
    points: int = 12,
) -> dict[str, Any]:
    if not records or not returns:
        return {"available": False, "as_of_date": str(as_of_date)[:10], "obs": 0}
    r = pd.Series(returns, index=pd.to_datetime([row["date"] for row in records]), dtype=float).fillna(0.0)
    eq = 1000.0 * np.exp(r.cumsum())
    out: dict[str, Any] = {
        "available": True,
        "as_of_date": str(as_of_date)[:10],
        "obs": int(len(r)),
        "metrics_full_period": _return_series_metrics(r),
        "equity_curve": _series_equity_snapshot(eq, points=points),
        "active_pair": records[-1].get("active_pair"),
        "current_regime": records[-1].get("portfolio_regime"),
        "recent_components": [
            {
                "date": str(row.get("date"))[:10],
                "active_pair": row.get("active_pair"),
                "portfolio_regime": row.get("portfolio_regime"),
                "return": row.get("returns_with_rf"),
                "raw_return": row.get("raw_returns_with_rf"),
                "switch_executed": row.get("switch_executed"),
            }
            for row in records[-10:]
        ],
    }
    out["metrics_by_window"] = {
        f"{int(window)}d": _return_series_metrics(r.tail(int(window)))
        for window in windows
    }
    out["equity_curve_by_window"] = {
        f"{int(window)}d": _series_equity_snapshot(eq.tail(int(window)), points=points)
        for window in windows
    }
    switch_dates = [pd.Timestamp(row["date"]) for row in records if row.get("switch_executed")]
    if switch_dates:
        out["switch_cadence"] = {
            "switch_count": int(len(switch_dates)),
            "days_since_last_switch": int((pd.Timestamp(as_of_date) - switch_dates[-1]).days),
            "last_switch_date": str(switch_dates[-1].date()),
        }
    else:
        out["switch_cadence"] = {"switch_count": 0, "days_since_last_switch": None, "last_switch_date": None}
    return out


def _return_series_metrics(returns: pd.Series) -> dict[str, Any]:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return {"obs": 0}
    eq = np.exp(r.cumsum())
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    return {
        "obs": int(len(r)),
        "total_return": float(eq.iloc[-1] - 1.0),
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
        "drawdown": float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0,
        "cvar_5": _cvar(r),
    }


def _series_equity_snapshot(equity: pd.Series, points: int = 12) -> list[dict[str, Any]]:
    s = pd.Series(equity).dropna().astype(float)
    if s.empty:
        return []
    if len(s) <= points:
        idx = np.arange(len(s))
    else:
        idx = np.unique(np.linspace(0, len(s) - 1, points).round().astype(int))
    return [
        {
            "date": str(s.index[i])[:10],
            "equity": float(s.iloc[i]),
            "return_from_start": float(s.iloc[i] / max(s.iloc[0], 1e-12) - 1.0),
        }
        for i in idx
    ]


def equity_curve_snapshot(frame: pd.DataFrame, end: int, points: int = 12, window: int | None = None) -> list[dict[str, Any]]:
    if window is None:
        hist = frame.iloc[: end + 1]
    else:
        hist = frame.iloc[max(0, end - window + 1) : end + 1]
    if hist.empty or "equity_curve_with_rf" not in hist:
        return []
    equity = hist["equity_curve_with_rf"].astype(float)
    if len(equity) <= points:
        idx = np.arange(len(equity))
    else:
        idx = np.unique(np.linspace(0, len(equity) - 1, points).round().astype(int))
    return [
        {
            "date": str(equity.index[i])[:10],
            "equity": float(equity.iloc[i]),
            "return_from_start": float(equity.iloc[i] / max(equity.iloc[0], 1e-12) - 1.0),
        }
        for i in idx
    ]


def build_past_regime_rankings(
    frames: dict[str, pd.DataFrame],
    end: int,
    window: int = 63,
    min_obs: int = 20,
    max_history: int = 252,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_regime: dict[str, list[dict[str, Any]]] = {
        regime: []
        for regime in PORTFOLIO_REGIMES
    }
    if end < min_obs:
        return rows_by_regime

    all_rows: list[dict[str, Any]] = []
    for pair, frame in frames.items():
        hist = frame.iloc[max(0, end - max_history + 1) : end + 1]
        if len(hist) < min_obs:
            continue
        r = hist["returns_with_rf"].fillna(0.0)
        eq = np.exp(r.cumsum())
        ann_ret = float(r.mean() * 252)
        ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        drawdown = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
        target = float(hist["target_vol"].dropna().iloc[-1]) if "target_vol" in hist and hist["target_vol"].notna().any() else 0.10
        row = {
            "pair": pair,
            "regime": "",
            "obs": int(len(r)),
            "annualized_return": ann_ret,
            "realized_vol": ann_vol,
            "vol_tracking_error": ann_vol - target,
            "rolling_sharpe": sharpe,
            "drawdown": drawdown,
            "cvar_5": _cvar(r),
            "turnover": float(hist.get("turnover", pd.Series(0.0, index=hist.index)).mean()),
        }
        row_regime = _regime_from_metrics(row)
        row["regime"] = row_regime
        rows_by_regime[row_regime].append(dict(row))
        all_rows.append(row)

    for regime, rows in rows_by_regime.items():
        if not rows:
            rows_by_regime[regime] = [dict(row, regime=regime, inferred_from_all_pairs=True) for row in all_rows]

    for regime, rows in rows_by_regime.items():
        rows.sort(key=_sharpe_tie_rank_key)
        for i, row in enumerate(rows, start=1):
            row["selection_rank"] = i
    return rows_by_regime


def build_ai_regime_rank_blocks(
    frames: dict[str, pd.DataFrame],
    end: int,
    regime_series: pd.Series | None,
    min_obs: int = 20,
    max_pairs_per_block: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Rank pairs inside prior AI-labeled regime date blocks only."""
    rank_rule = "Sharpe first; within a 0.05 Sharpe tie band, drawdown second, turnover third, cVaR fourth"
    blocks: dict[str, dict[str, Any]] = {
        regime: {
            "regime": regime,
            "rank_definition": "Pair rank among candidates using only past dates labeled as this AI portfolio regime.",
            "rank_importance": "Important input for switch_review and choose_pair; not a whole-history rank.",
            "rank_rule": rank_rule,
            "obs_dates": 0,
            "pairs": [],
        }
        for regime in PORTFOLIO_REGIMES
    }
    if regime_series is None or end < 1:
        return blocks

    past_regimes = regime_series.iloc[: end + 1].dropna().astype(str)
    if past_regimes.empty:
        return blocks

    for regime in blocks:
        regime_dates = past_regimes[past_regimes == regime].index
        blocks[regime]["obs_dates"] = int(len(regime_dates))
        if len(regime_dates) < min_obs:
            blocks[regime]["insufficient_obs"] = True
            continue

        rows: list[dict[str, Any]] = []
        for pair, frame in frames.items():
            hist = frame.loc[frame.index.intersection(regime_dates)]
            if len(hist) < min_obs:
                continue
            r = hist["returns_with_rf"].fillna(0.0)
            eq = np.exp(r.cumsum())
            ann_ret = float(r.mean() * 252)
            ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            drawdown = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
            if "target_vol" in hist and hist["target_vol"].notna().any():
                target = float(hist["target_vol"].dropna().median())
            else:
                target = 0.10
            rows.append(
                {
                    "pair": pair,
                    "regime": regime,
                    "obs": int(len(hist)),
                    "annualized_return": ann_ret,
                    "realized_vol": ann_vol,
                    "vol_tracking_error": ann_vol - target,
                    "rolling_sharpe": sharpe,
                    "drawdown": drawdown,
                    "cvar_5": _cvar(r),
                    "turnover": float(hist.get("turnover", pd.Series(0.0, index=hist.index)).mean()),
                }
            )

        rows.sort(key=_sharpe_tie_rank_key)
        for i, row in enumerate(rows, start=1):
            row["selection_rank"] = i
        if max_pairs_per_block:
            rows = rows[:max_pairs_per_block]
        blocks[regime]["pairs"] = rows
        blocks[regime]["insufficient_obs"] = False
    return blocks


def build_recent_overall_rankings(
    frames: dict[str, pd.DataFrame],
    end: int,
    lookback: int = 60,
    min_obs: int = 20,
    tie_band: float = 0.05,
    max_pairs: int | None = None,
) -> list[dict[str, Any]]:
    """Rank all pairs on the recent lookback window without using regime labels."""
    rows: list[dict[str, Any]] = []
    if end < min_obs:
        return rows
    for pair, frame in frames.items():
        hist = frame.iloc[max(0, end - lookback + 1) : end + 1]
        if len(hist) < min_obs:
            continue
        r = hist["returns_with_rf"].fillna(0.0)
        eq = np.exp(r.cumsum())
        ann_ret = float(r.mean() * 252)
        ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        drawdown = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
        target = float(hist["target_vol"].dropna().iloc[-1]) if "target_vol" in hist and hist["target_vol"].notna().any() else 0.10
        rows.append(
            {
                "pair": pair,
                "obs": int(len(hist)),
                "lookback_days": int(lookback),
                "annualized_return": ann_ret,
                "realized_vol": ann_vol,
                "vol_tracking_error": ann_vol - target,
                "rolling_sharpe": sharpe,
                "drawdown": drawdown,
                "cvar_5": _cvar(r),
                "turnover": float(hist.get("turnover", pd.Series(0.0, index=hist.index)).mean()),
                "rank_rule": f"Sharpe first; within {tie_band:.3f} Sharpe tie band, drawdown, turnover, cVaR",
            }
        )
    rows.sort(key=lambda row: _sharpe_tie_rank_key(row, tie_band=tie_band))
    for i, row in enumerate(rows, start=1):
        row["overall_rank"] = i
    if max_pairs:
        rows = rows[:max_pairs]
    return rows


def load_pair_results(manifest_path: str | Path) -> dict[str, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    frames: dict[str, pd.DataFrame] = {}
    for row in manifest.to_dict("records"):
        if row.get("status") != "ok":
            continue
        name = str(row["name"])
        if "__mean_variance" in name or "buy_and_hold" in name:
            continue
        path = Path(str(row.get("path", "")))
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        frames[name] = df.sort_index()
    return frames


def _filter_pair_results(frames: dict[str, pd.DataFrame], params: dict[str, Any]) -> dict[str, pd.DataFrame]:
    include_pairs = params.get("included_pairs") or params.get("allowed_pairs")
    if isinstance(include_pairs, str):
        include_pairs = [x.strip() for x in include_pairs.split(",") if x.strip()]
    if include_pairs:
        allowed = {str(pair) for pair in include_pairs}
        frames = {name: frame for name, frame in frames.items() if name in allowed}
    tokens = params.get("excluded_pairs_containing", [])
    if isinstance(tokens, str):
        tokens = [tokens]
    tokens = [str(token).lower() for token in tokens if str(token)]
    if not tokens:
        return frames
    return {
        name: frame
        for name, frame in frames.items()
        if not any(token in name.lower() for token in tokens)
    }


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        found = cols.get(candidate.lower())
        if found is not None:
            return found
    return None


def _load_vol_regime_inputs(params: dict[str, Any]) -> pd.DataFrame:
    returns_path = Path(str(params.get("multi_asset_returns_path", "data/processed/emm_daily_log_returns_yahoo_20220210_20260210.parquet")))
    vix_path = Path(str(params.get("vix_path", "data/processed/VIX_Daily_Processed.parquet")))
    master_path = Path(str(params.get("termspread_path", "data/processed/Master_Dataset.parquet")))
    credit_path = Path(str(params.get("credit_spread_path", "data/processed/credit_spreads_2021_2026.parquet")))
    paths = [returns_path, vix_path, master_path, credit_path]
    paths = [p if p.is_absolute() else Path.cwd() / p for p in paths]
    returns_path, vix_path, master_path, credit_path = paths

    returns = pd.read_parquet(returns_path).sort_index()
    returns.index = pd.to_datetime(returns.index)
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns.columns = [f"asset_return_{c}" for c in returns.columns]
    out = returns.copy()
    asset_cols = list(out.columns)
    out["equal_weight_return"] = out[asset_cols].mean(axis=1)
    out["equal_weight_realized_vol_21d"] = out["equal_weight_return"].rolling(21, min_periods=10).std() * np.sqrt(252)
    out["equal_weight_realized_vol_63d"] = out["equal_weight_return"].rolling(63, min_periods=20).std() * np.sqrt(252)

    vix_df = pd.read_parquet(vix_path).sort_index()
    vix_df.index = pd.to_datetime(vix_df.index)
    vix_col = _first_existing_col(vix_df, ["vix_close", "vix", "close", "value", "adj_close", "price"])
    if vix_col is not None:
        out["vix"] = pd.to_numeric(vix_df[vix_col], errors="coerce").reindex(out.index).ffill()

    master = pd.read_parquet(master_path).sort_index()
    master.index = pd.to_datetime(master.index)
    term_col = _first_existing_col(master, [str(params.get("termspread_col", "t10y2y")), "termspread", "term_spread", "t10y2y"])
    if term_col is not None:
        out["termspread"] = pd.to_numeric(master[term_col], errors="coerce").reindex(out.index).ffill()

    credit = pd.read_parquet(credit_path).sort_index()
    credit.index = pd.to_datetime(credit.index)
    for col in ["hy_oas", "ig_oas", "aaa_oas", "bbb_oas", "hy_oas_change", "ig_oas_change", "aaa_oas_change", "bbb_oas_change"]:
        if col in credit.columns:
            out[f"credit_{col}"] = pd.to_numeric(credit[col], errors="coerce").reindex(out.index).ffill()

    return out


def _series_summary(series: pd.Series, windows: tuple[int, ...] = (20, 63, 126)) -> dict[str, Any]:
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return {"available": False}
    out: dict[str, Any] = {"available": True, "latest": float(s.iloc[-1])}
    for window in windows:
        w = s.tail(window)
        if len(w) == 0:
            continue
        out[f"mean_{window}d"] = float(w.mean())
        out[f"change_{window}d"] = float(w.iloc[-1] - w.iloc[0]) if len(w) > 1 else 0.0
        out[f"min_{window}d"] = float(w.min())
        out[f"max_{window}d"] = float(w.max())
    return out


def _vol_regime_payload(input_frame: pd.DataFrame, end: int, date: Any) -> dict[str, Any]:
    hist = input_frame.iloc[: max(end, -1) + 1]
    asset_cols = [c for c in hist.columns if c.startswith("asset_return_")]
    asset_tail = hist[asset_cols].tail(63) if asset_cols else pd.DataFrame(index=hist.index)
    asset_summary = {}
    for col in asset_cols:
        s = hist[col].dropna().astype(float)
        w = s.tail(63)
        asset_summary[col.replace("asset_return_", "")] = {
            "latest_return": float(s.iloc[-1]) if len(s) else None,
            "return_20d": float(s.tail(20).sum()) if len(s) else None,
            "return_63d": float(w.sum()) if len(w) else None,
            "realized_vol_63d": float(w.std(ddof=1) * np.sqrt(252)) if len(w) > 1 else None,
            "max_drawdown_63d": _simple_drawdown(w),
        }
    corr_level = None
    if len(asset_tail) > 20 and asset_tail.shape[1] > 1:
        corr = asset_tail.corr().values
        upper = corr[np.triu_indices_from(corr, k=1)]
        corr_level = float(np.nanmean(upper)) if len(upper) else None

    return {
        "instruction": "Classify the current multi-asset volatility regime using only these supplied series through as_of_date. Do not infer from same-day or future observations.",
        "date": str(date)[:10],
        "as_of_date": str(hist.index[-1])[:10] if len(hist) else None,
        "allowed_regimes": list(PORTFOLIO_REGIMES),
        "regime_meaning": {
            "risk_on": "benign or low volatility",
            "balanced": "normal or mixed volatility",
            "defensive": "elevated or rising volatility/stress",
        },
        "data_restriction": "No strategy pair metrics, pair ranks, selected strategy returns, same-day routed return, or future observations are included or allowed.",
        "multi_asset_individual_returns": asset_summary,
        "cross_asset_return_correlation_63d": corr_level,
        "equal_weight_portfolio": {
            "return": _series_summary(hist.get("equal_weight_return", pd.Series(dtype=float))),
            "realized_vol_21d": _series_summary(hist.get("equal_weight_realized_vol_21d", pd.Series(dtype=float))),
            "realized_vol_63d": _series_summary(hist.get("equal_weight_realized_vol_63d", pd.Series(dtype=float))),
        },
        "vix": _series_summary(hist.get("vix", pd.Series(dtype=float))),
        "termspread": _series_summary(hist.get("termspread", pd.Series(dtype=float))),
        "credit_spreads": {
            col.replace("credit_", ""): _series_summary(hist[col])
            for col in hist.columns
            if col.startswith("credit_")
        },
    }


def _simple_drawdown(returns: pd.Series) -> float | None:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return None
    eq = np.exp(r.cumsum())
    return float((eq / eq.cummax() - 1.0).min())


def _fallback_volatility_regime(payload: dict[str, Any]) -> dict[str, Any]:
    ew_vol = payload.get("equal_weight_portfolio", {}).get("realized_vol_63d", {})
    vix = payload.get("vix", {})
    credit = payload.get("credit_spreads", {})
    vol_latest = float(ew_vol.get("latest", 0.0) or 0.0)
    vix_latest = float(vix.get("latest", 0.0) or 0.0)
    hy = credit.get("hy_oas", {})
    hy_latest = float(hy.get("latest", 0.0) or 0.0) if hy.get("available") else 0.0
    hy_change = float(hy.get("change_20d", 0.0) or 0.0) if hy.get("available") else 0.0
    if vol_latest >= 0.14 or vix_latest >= 24 or hy_latest >= 5.0 or hy_change >= 0.50:
        regime = "defensive"
    elif vol_latest <= 0.09 and vix_latest <= 18 and (not hy.get("available") or hy_change <= 0.15):
        regime = "risk_on"
    else:
        regime = "balanced"
    return {"portfolio_regime": regime, "confidence": 0.0, "reason": "Deterministic volatility input fallback"}


def _write_router_progress(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    progress_path = Path(path)
    if not progress_path.is_absolute():
        progress_path = Path.cwd() / progress_path
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True), encoding="utf-8")


def run_ai_portfolio_router(
    manifest_path: str | Path,
    out_path: str | Path,
    params: dict[str, Any] | None = None,
    window: int = 63,
    regime_rank_window: int | None = None,
    start_date: str | None = None,
    last_years: float | None = None,
) -> pd.DataFrame:
    load_dotenv()
    params = dict(params or {})
    progress_path = params.get("progress_path")
    _write_router_progress(
        progress_path,
        {
            "phase": "initializing",
            "manifest_path": str(manifest_path),
            "out_path": str(out_path),
            "start_date": start_date,
        },
    )
    frames = load_pair_results(manifest_path)
    frames = _filter_pair_results(frames, params)
    if not frames:
        raise ValueError(f"No successful pair parquet files found from manifest: {manifest_path}")
    common_index = sorted(set.intersection(*[set(df.index) for df in frames.values()]))
    if not common_index:
        raise ValueError("Pair result files have no common dates.")
    common_index = pd.DatetimeIndex(common_index)
    metric_start_date = params.get("metric_start_date", params.get("history_start_date"))
    if last_years is not None:
        cutoff = common_index.max() - pd.DateOffset(days=int(float(last_years) * 365.25))
        common_index = common_index[common_index >= cutoff]
    if metric_start_date:
        common_index = common_index[common_index >= pd.Timestamp(str(metric_start_date))]
    route_index = common_index
    if start_date:
        route_index = common_index[common_index >= pd.Timestamp(start_date)]
    if len(common_index) == 0 or len(route_index) == 0:
        raise ValueError("No common dates remain after start-date/last-years filtering.")
    frames = {name: df.loc[common_index].copy() for name, df in frames.items()}
    router = AIPortfolioRegimeRouter(params)
    requested_initial_pair = params.get("initial_pair")
    if requested_initial_pair in frames:
        active_pair = str(requested_initial_pair)
    else:
        active_pair = sorted(frames)[0]
    active_pair_start_route_step = 0
    records: list[dict[str, Any]] = []
    router_returns = []
    regime_rank_window = int(regime_rank_window or window)
    precomputed_regimes = load_precomputed_regime_series(params.get("precomputed_regime_path"))
    precomputed_regime_series = pd.Series(
        [
            precomputed_regimes.get(str(date)[:10], {}).get("portfolio_regime")
            for date in common_index
        ],
        index=common_index,
        dtype="object",
    )
    last_regime_rankings: dict[str, Any] = build_ai_regime_rank_blocks(frames, 0, None)
    last_overall_rankings: list[dict[str, Any]] = []
    last_recent_rankings: dict[int, list[dict[str, Any]]] = {}
    switch_log: list[dict[str, Any]] = []
    previous_loaded_regime: str | None = None

    for route_step, date in enumerate(route_index):
        progress_base = {
            "route_step": int(route_step),
            "route_rows": int(len(route_index)),
            "date": str(date)[:10],
            "active_pair": active_pair,
            "records_completed": int(len(records)),
            "switches_completed": int(len(switch_log)),
        }
        _write_router_progress(progress_path, {**progress_base, "phase": "route_step_start"})
        step = int(common_index.get_loc(date))
        signal_step = step - 1
        signal_as_of_date = common_index[signal_step] if signal_step >= 0 else None
        all_metrics = {name: rolling_metrics(df, signal_step, window=window) for name, df in frames.items()}
        pair_summaries = {name: pair_summary_metrics(df, signal_step) for name, df in frames.items()}
        pair_window_summaries = {
            name: {
                f"{int(rank_window)}d": pair_window_metrics(df, signal_step, int(rank_window))
                for rank_window in router.recent_rank_windows
            }
            for name, df in frames.items()
        }
        pair_drawdown_profiles = {
            name: drawdown_consistency_profile(
                pair_window_summaries.get(name, {}),
                target_abs_drawdown=router.drawdown_consistency_target,
                max_abs_drawdown_target=router.max_drawdown_target,
            )
            for name in frames
        }
        for name, profile in pair_drawdown_profiles.items():
            all_metrics.setdefault(name, {})["drawdown_consistency_score"] = float(
                profile.get("consistency_score", -np.inf)
            )
            all_metrics[name]["drawdown_consistency_mean_abs"] = profile.get("mean_abs_drawdown")
            all_metrics[name]["drawdown_consistency_std_abs"] = profile.get("std_abs_drawdown")
            all_metrics[name]["drawdown_consistency_target_gap"] = profile.get("mean_gap_to_target")
            all_metrics[name]["max_abs_drawdown"] = profile.get("max_abs_drawdown")
            all_metrics[name]["max_drawdown_target_gap"] = profile.get("max_drawdown_target_gap")
            all_metrics[name]["max_drawdown_excess_over_target"] = profile.get("max_drawdown_excess_over_target")
        equity_curves = {
            name: equity_curve_snapshot(df, signal_step, points=router.equity_snapshot_points)
            for name, df in frames.items()
        }
        equity_curves_by_window = {
            int(rank_window): {
                name: equity_curve_snapshot(
                    df,
                    signal_step,
                    points=router.equity_snapshot_points,
                    window=int(rank_window),
                )
                for name, df in frames.items()
            }
            for rank_window in router.recent_rank_windows
        } if router.include_equity_curves_by_window else {}
        review_due = route_step == 0 or route_step - router._last_switch_review_step >= router.switch_review_interval
        regime_key = str(date)[:10]
        loaded_regime_name = None
        if regime_key in precomputed_regimes:
            loaded_regime_name = str(precomputed_regimes[regime_key].get("portfolio_regime", ""))
        if review_due or step == 0:
            if precomputed_regimes:
                last_regime_rankings = build_ai_regime_rank_blocks(
                    frames,
                    signal_step,
                    precomputed_regime_series,
                    min_obs=max(20, min(regime_rank_window, 63) // 2),
                    max_pairs_per_block=max(router.regime_rank_top_n * 2, router.candidate_top_n * 2),
                )
            else:
                last_regime_rankings = build_past_regime_rankings(
                    frames,
                    signal_step,
                    window=regime_rank_window,
                    min_obs=max(20, min(window, 63) // 2),
                )
            last_overall_rankings = build_recent_overall_rankings(
                frames,
                signal_step,
                lookback=router.overall_rank_window,
                min_obs=max(20, min(router.overall_rank_window, 60) // 2),
                tie_band=router.sharpe_rank_tie_band,
                max_pairs=max(router.overall_rank_top_k * 2, router.candidate_top_n * 2),
            )
            last_recent_rankings = {
                int(rank_window): build_recent_overall_rankings(
                    frames,
                    signal_step,
                    lookback=int(rank_window),
                    min_obs=max(5, min(int(rank_window), 60) // 2),
                    tie_band=router.sharpe_rank_tie_band,
                    max_pairs=None,
                )
                for rank_window in router.recent_rank_windows
            }
            router.observe_rankings(last_regime_rankings, route_step, date=date)
        regime_rankings = last_regime_rankings
        overall_rankings = last_overall_rankings
        recent_rankings = last_recent_rankings
        for regime_name in PORTFOLIO_REGIMES:
            for row in router._rank_candidates(all_metrics, regime_name):
                all_metrics[row["pair"]][f"{regime_name}_score"] = row["score"]
        active_metrics = all_metrics.get(active_pair, {})
        if regime_key in precomputed_regimes:
            regime = dict(precomputed_regimes[regime_key])
            previous_loaded_regime = str(regime.get("portfolio_regime", ""))
            router._last_regime = dict(regime)
            router._last_regime_step = route_step
            if not router._regime_history or router._regime_history[-1].get("date") != regime_key:
                router._regime_history.append(
                    {
                        "step": route_step,
                        "date": regime_key,
                        "portfolio_regime": regime.get("portfolio_regime"),
                        "confidence": regime.get("confidence", 0.0),
                        "reason": regime.get("reason", ""),
                    }
                )
        else:
            regime = router.classify_regime(
                route_step,
                active_pair=active_pair,
                active_metrics=active_metrics,
                all_metrics=all_metrics,
                regime_rankings=regime_rankings,
                date=date,
            )
        strategy_context = routed_strategy_context(
            records,
            router_returns,
            as_of_date=date,
            windows=router.recent_rank_windows,
        )
        active_hold_days = route_step - active_pair_start_route_step + 1
        if router.initial_hold_days and route_step < router.initial_hold_days:
            switch_decision = {
                "action": "hold",
                "reason": f"Initial training-selected pair hold {route_step + 1}/{router.initial_hold_days}",
                "confidence": 0.0,
                "regime_changed": False,
                "previous_portfolio_regime": router._last_portfolio_regime,
                "initial_hold_active": True,
            }
        else:
            _write_router_progress(
                progress_path,
                {
                    **progress_base,
                    "phase": "before_switch_review",
                    "signal_as_of_date": str(signal_as_of_date)[:10] if signal_as_of_date is not None else None,
                    "portfolio_regime": regime.get("portfolio_regime", "balanced"),
                    "review_due": bool(review_due),
                    "active_hold_days": int(active_hold_days),
                },
            )
            switch_decision = router.switch_review(
                route_step,
                active_pair,
                active_metrics,
                all_metrics,
                regime,
                regime_rankings=regime_rankings,
                overall_rankings=overall_rankings,
                recent_rankings=recent_rankings,
                switch_log=switch_log,
                strategy_context=strategy_context,
                active_hold_days=active_hold_days,
                pair_summaries=pair_summaries,
                pair_window_summaries=pair_window_summaries,
                pair_drawdown_profiles=pair_drawdown_profiles,
                equity_curves=equity_curves,
                equity_curves_by_window=equity_curves_by_window,
                date=date,
            )
            _write_router_progress(
                progress_path,
                {
                    **progress_base,
                    "phase": "after_switch_review",
                    "signal_as_of_date": str(signal_as_of_date)[:10] if signal_as_of_date is not None else None,
                    "switch_review_action": switch_decision.get("action", "hold"),
                    "switch_review_error": switch_decision.get("error", ""),
                    "switch_review_reason": switch_decision.get("reason", ""),
                },
            )
        if switch_decision.get("action") == "switch":
            _write_router_progress(
                progress_path,
                {
                    **progress_base,
                    "phase": "before_choose_pair",
                    "signal_as_of_date": str(signal_as_of_date)[:10] if signal_as_of_date is not None else None,
                    "portfolio_regime": regime.get("portfolio_regime", "balanced"),
                },
            )
            decision = router.choose_pair(
                route_step,
                active_pair,
                active_metrics,
                all_metrics,
                regime,
                regime_rankings=regime_rankings,
                overall_rankings=overall_rankings,
                recent_rankings=recent_rankings,
                switch_log=switch_log,
                pair_summaries=pair_summaries,
                pair_window_summaries=pair_window_summaries,
                pair_drawdown_profiles=pair_drawdown_profiles,
                equity_curves=equity_curves,
                equity_curves_by_window=equity_curves_by_window,
                date=date,
            )
            _write_router_progress(
                progress_path,
                {
                    **progress_base,
                    "phase": "after_choose_pair",
                    "signal_as_of_date": str(signal_as_of_date)[:10] if signal_as_of_date is not None else None,
                    "decision_action": decision.get("action", "hold"),
                    "decision_pair": decision.get("pair", active_pair),
                    "decision_error": decision.get("error", ""),
                },
            )
        else:
            decision = {
                "action": "hold",
                "pair": active_pair,
                "reason": switch_decision.get("reason", "Switch review held active pair"),
                "confidence": switch_decision.get("confidence", 0.0),
            }
        prior_active_pair = active_pair
        switch_executed = False
        if decision.get("action") == "switch" and decision.get("pair") in frames:
            active_pair = str(decision["pair"])
            switch_executed = active_pair != prior_active_pair
            if switch_executed:
                active_pair_start_route_step = route_step
        raw_ret = float(frames[active_pair].loc[date, "returns_with_rf"])
        overlay_leverage = 1.0
        switch_cost_tracked = float(router.switch_cost_penalty if switch_executed else 0.0)
        switch_penalty_applied = switch_cost_tracked if router.apply_switch_penalty_to_returns else 0.0
        ret = raw_ret - switch_penalty_applied
        if switch_executed:
            switch_log.append(
                {
                    "date": str(date)[:10],
                    "from_pair": prior_active_pair,
                    "to_pair": active_pair,
                    "portfolio_regime": regime.get("portfolio_regime", "balanced"),
                    "previous_portfolio_regime": switch_decision.get("previous_portfolio_regime"),
                    "switch_cost_tracked_not_applied": switch_cost_tracked if not router.apply_switch_penalty_to_returns else 0.0,
                }
            )
        router_returns.append(ret)
        records.append(
            {
                "date": date,
                "router_signal_as_of_date": str(signal_as_of_date)[:10] if signal_as_of_date is not None else None,
                "active_pair": active_pair,
                "portfolio_regime": regime.get("portfolio_regime", "balanced"),
                "regime_confidence": regime.get("confidence", 0.0),
                "regime_reason": regime.get("reason", ""),
                "decision_action": decision.get("action", "hold"),
                "decision_pair": decision.get("pair", active_pair),
                "decision_reason": decision.get("reason", ""),
                "decision_error": decision.get("error", ""),
                "selection_ai_success": bool(
                    switch_decision.get("action") == "switch"
                    and not decision.get("error")
                    and decision.get("pair") in frames
                ),
                "switch_executed": switch_executed,
                "switch_penalty_applied": switch_penalty_applied,
                "switch_cost_tracked": switch_cost_tracked,
                "switch_penalty_return_applied": router.apply_switch_penalty_to_returns,
                "switch_review_action": switch_decision.get("action", "hold"),
                "switch_review_reason": switch_decision.get("reason", ""),
                "switch_review_error": switch_decision.get("error", ""),
                "switch_review_ai_success": bool(
                    switch_decision.get("action") in {"hold", "switch"}
                    and not switch_decision.get("error")
                    and not str(switch_decision.get("reason", "")).startswith(("Switch review interval hold", "No candidate available", "Cooldown hold", "Initial training-selected pair hold"))
                ),
                "regime_changed": switch_decision.get("regime_changed", False),
                "previous_portfolio_regime": switch_decision.get("previous_portfolio_regime"),
                "cooldown_active": switch_decision.get("cooldown_active", False),
                "hold_gate_active": switch_decision.get(
                    "hold_gate_active",
                    switch_decision.get("cooldown_active", False),
                ),
                "active_hold_days": active_hold_days,
                "cooldown_ranks": json.dumps(switch_decision.get("cooldown_ranks", {}), sort_keys=True),
                "cooldown_poor_flags": json.dumps(switch_decision.get("cooldown_poor_flags", {}), sort_keys=True),
                "cooldown_all_poor": switch_decision.get("cooldown_all_poor"),
                "sensitivity_enabled": router.sensitivity_enabled,
                "sensitivity": router.sensitivity,
                "sensitivity_hold_active": switch_decision.get("sensitivity_hold_active", False),
                "sensitivity_group_check": json.dumps(
                    switch_decision.get("sensitivity_group_check", {}),
                    sort_keys=True,
                ),
                "initial_hold_active": switch_decision.get("initial_hold_active", False),
                "active_regime_rank": next(
                    (
                        row.get("selection_rank")
                        for row in _rank_rows_for_regime(regime_rankings, str(regime.get("portfolio_regime", "balanced")))
                        if row.get("pair") == active_pair
                    ),
                    None,
                ),
                "active_overall_recent_rank": next(
                    (
                        row.get("overall_rank")
                        for row in overall_rankings
                        if row.get("pair") == active_pair
                    ),
                    None,
                ),
                "raw_returns_with_rf": raw_ret,
                "router_vol_overlay_leverage": overlay_leverage,
                "returns_with_rf": ret,
                "equity_curve_with_rf": float(1000.0 * np.exp(np.sum(router_returns))),
                **{f"active_{k}": v for k, v in active_metrics.items()},
            }
        )
        _write_router_progress(
            progress_path,
            {
                **progress_base,
                "phase": "record_appended",
                "signal_as_of_date": str(signal_as_of_date)[:10] if signal_as_of_date is not None else None,
                "active_pair_after_decision": active_pair,
                "switch_executed": bool(switch_executed),
                "records_completed": int(len(records)),
                "switches_completed": int(len(switch_log)),
            },
        )
    out = pd.DataFrame(records).set_index("date")
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    out.to_csv(out_path.with_suffix(".csv"))
    _write_router_progress(
        progress_path,
        {
            "phase": "complete",
            "out_path": str(out_path),
            "rows": int(len(out)),
            "switches_completed": int(len(switch_log)),
        },
    )
    return out


def load_precomputed_regime_series(path_value: Any) -> dict[str, dict[str, Any]]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        date = str(row.get("date", ""))[:10]
        regime = str(row.get("portfolio_regime", row.get("regime", "balanced")))
        if date and regime in set(PORTFOLIO_REGIMES):
            out[date] = {
                "portfolio_regime": regime,
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "reason": str(row.get("reason", "")),
                "source": "precomputed",
            }
    return out


def generate_ai_portfolio_regime_series(
    manifest_path: str | Path,
    out_path: str | Path,
    params: dict[str, Any] | None = None,
    window: int = 63,
    regime_rank_window: int | None = None,
    interval: int = 20,
    start_date: str | None = None,
    last_years: float | None = None,
) -> pd.DataFrame:
    load_dotenv()
    params = dict(params or {})
    regime_input_mode = str(params.get("regime_input_mode", "pair_metrics")).lower()
    if regime_input_mode in {"macro_vol", "volatility", "volatility_inputs"}:
        input_frame = _load_vol_regime_inputs(params)
        full_index = pd.DatetimeIndex(input_frame.index)
        common_index = full_index
        frames: dict[str, pd.DataFrame] = {}
    else:
        frames = load_pair_results(manifest_path)
        if not frames:
            raise ValueError(f"No successful pair parquet files found from manifest: {manifest_path}")
        full_index = pd.DatetimeIndex(sorted(set.intersection(*[set(df.index) for df in frames.values()])))
        common_index = full_index
        input_frame = pd.DataFrame(index=common_index)
    if last_years is not None:
        cutoff = common_index.max() - pd.DateOffset(days=int(float(last_years) * 365.25))
        common_index = common_index[common_index >= cutoff]
    if start_date:
        common_index = common_index[common_index >= pd.Timestamp(start_date)]
    if len(common_index) == 0:
        raise ValueError("No common dates remain after start-date/last-years filtering.")
    if regime_input_mode not in {"macro_vol", "volatility", "volatility_inputs"}:
        frames = {name: df.loc[full_index].copy() for name, df in frames.items()}
    router = AIPortfolioRegimeRouter(params)
    active_pair = sorted(frames)[0] if frames else "macro_volatility_inputs_only"
    regime_rank_window = int(regime_rank_window or window)
    interval = max(int(interval), 1)
    last_regime: dict[str, Any] | None = None
    last_regime_rankings: dict[str, list[dict[str, Any]]] = {regime: [] for regime in PORTFOLIO_REGIMES}
    records: list[dict[str, Any]] = []

    for step, date in enumerate(common_index):
        full_step = int(full_index.get_loc(date))
        signal_step = full_step - 1
        signal_as_of_date = full_index[signal_step] if signal_step >= 0 else None
        should_call = last_regime is None or step % interval == 0
        if should_call:
            if regime_input_mode in {"macro_vol", "volatility", "volatility_inputs"}:
                payload = _vol_regime_payload(input_frame, signal_step, date)
                try:
                    if not router.ai_enabled:
                        raise ValueError("AI disabled")
                    regime = router._call_ai(json.dumps(payload, default=str), AI_VOLATILITY_REGIME_PROMPT)
                    if regime.get("portfolio_regime") not in set(PORTFOLIO_REGIMES):
                        raise ValueError("Invalid volatility regime")
                except Exception as exc:
                    regime = _fallback_volatility_regime(payload)
                    regime["error"] = str(exc)
                router._regime_history.append(
                    {
                        "step": step,
                        "date": str(date)[:10],
                        "portfolio_regime": regime.get("portfolio_regime"),
                        "confidence": regime.get("confidence", 0.0),
                        "reason": regime.get("reason", ""),
                    }
                )
            else:
                all_metrics = {name: rolling_metrics(df, signal_step, window=window) for name, df in frames.items()}
                active_metrics = all_metrics.get(active_pair, {})
                last_regime_rankings = build_past_regime_rankings(
                    frames,
                    signal_step,
                    window=regime_rank_window,
                    min_obs=max(20, min(window, 63) // 2),
                )
                regime = router.classify_regime(
                    step,
                    active_pair=active_pair,
                    active_metrics=active_metrics,
                    all_metrics=all_metrics,
                    regime_rankings=last_regime_rankings,
                    date=date,
                )
            regime["call_made"] = True
            last_regime = dict(regime)
        else:
            regime = dict(last_regime or {"portfolio_regime": "balanced", "confidence": 0.0, "reason": "initial fallback"})
            regime["call_made"] = False
        records.append(
            {
                "date": date,
                "regime_signal_as_of_date": str(signal_as_of_date)[:10] if signal_as_of_date is not None else None,
                "portfolio_regime": regime.get("portfolio_regime", "balanced"),
                "confidence": regime.get("confidence", 0.0),
                "reason": regime.get("reason", ""),
                "call_made": bool(regime.get("call_made", False)),
                "source": regime.get("source", "ai" if regime.get("call_made") else "carried"),
                "active_pair_context": active_pair,
            }
        )

    out = pd.DataFrame(records)
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    out.to_parquet(out_path.with_suffix(".parquet"), index=False)
    return out
