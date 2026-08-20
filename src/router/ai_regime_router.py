"""AI regime router with Week 10-style regime-conditioned candidate ranking.

At each router step this class can ask AI to predict the current volatility
regime. It then looks up historical pair behavior inside that regime and asks
AI whether to hold or switch using an unranked pair metric pool.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.router.router import Router
from src.router.strategy_pair import StrategyPair


def _post_json_worker(
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
    queue: Any,
) -> None:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        request_timeout = None if timeout <= 0 else timeout
        with urlopen(request, timeout=request_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        queue.put(("ok", data))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        queue.put(("error", f"HTTP {exc.code}: {detail}"))
    except URLError as exc:
        queue.put(("error", str(exc.reason)))
    except Exception as exc:
        queue.put(("error", str(exc)))


AI_REGIME_SYSTEM_PROMPT = """Output JSON immediately. Do not explain. Do not reason step by step.
You generate regime information for a volatility-targeting router.
Classify the current market into exactly one volatility regime: low, middle, or high.
Use only the supplied past information in the payload. Prefer persistent realized volatility,
vol-target error, drawdown, benchmark context, and recent regime history over one-day noise.
Do not use future strategy returns, future pair rankings, or any data outside the payload.
Return only:
{"vol_regime": "low", "confidence": 0.0, "reason": "<short reason>"}
or:
{"vol_regime": "middle", "confidence": 0.0, "reason": "<short reason>"}
or:
{"vol_regime": "high", "confidence": 0.0, "reason": "<short reason>"}
Keep reason under 24 words.
"""


AI_REGIME_SELECTION_SYSTEM_PROMPT = """Return JSON only. No reasoning. No prose.
Choose exactly one supplied pair name. The active pair is not supplied and cannot be chosen.
Use the candidate rows only. Lower selection_rank is better; use turnover, drawdown, and Sharpe only as tie-breakers.
Return exactly:
{"pair":"<supplied pair name>"}
"""


AI_REGIME_SWITCH_SYSTEM_PROMPT = """Return JSON only. No reasoning. No prose.
Decide hold or switch. Regime/context change or weak active recent rank favors switch.
Strong active rank and low drawdown favors hold.
Return exactly one of:
{"action":"hold"}
or
{"action":"switch"}
"""


class AIRegimeRouter(Router):
    """Router that combines AI regime prediction with regime-conditioned selection."""

    responses_endpoint = "https://api.openai.com/v1/responses"
    chat_completions_endpoint = "https://api.openai.com/v1/chat/completions"
    nvidia_endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, pairs: list[StrategyPair], params: dict[str, Any] | None = None):
        params = dict(params or {})
        params.setdefault("regime_bias_weight", 2.5)
        params.setdefault("regime_suitability_scale", 1.0)
        super().__init__(pairs=pairs, params=params)

        p = self.params
        self.provider = str(p.get("provider", "nvidia")).lower()
        self.api_format = str(p.get("api_format", "chat_completions")).lower()
        self.model = str(p.get("model", "openai/gpt-oss-120b"))
        api_key_param = str(p.get("api_key", ""))
        if api_key_param.startswith("${") and api_key_param.endswith("}"):
            api_key_param = os.getenv(api_key_param[2:-1], "")
        elif api_key_param.startswith("$"):
            api_key_param = os.getenv(api_key_param[1:], "")
        self.api_key = str(
            api_key_param
            or os.getenv("OPENAI_API_KEY", "")
            or os.getenv("NVIDIA_API_KEY", "")
            or os.getenv("NVAPI_KEY", "")
        )
        self.endpoint = str(p.get("endpoint", "") or self._default_endpoint())
        self.system_prompt = str(p.get("ai_regime_system_prompt", AI_REGIME_SYSTEM_PROMPT))
        self.selection_system_prompt = str(
            p.get("ai_selection_system_prompt", AI_REGIME_SELECTION_SYSTEM_PROMPT)
        )
        self.switch_system_prompt = str(
            p.get("ai_switch_system_prompt", AI_REGIME_SWITCH_SYSTEM_PROMPT)
        )
        self.max_output_tokens = int(p.get("max_output_tokens", 256))
        raw_temperature = p.get("temperature", 0.0)
        self.temperature = None if raw_temperature is None else float(raw_temperature)
        self.reasoning_effort = p.get("reasoning_effort")
        self.timeout = float(p.get("timeout", 45.0))
        self.request_min_interval_seconds = max(
            float(p.get("request_min_interval_seconds", 0.0)),
            0.0,
        )
        self._last_ai_request_monotonic = 0.0
        self.response_format = p.get("response_format")
        self.switch_decision_max_output_tokens = int(
            p.get("switch_decision_max_output_tokens", self.max_output_tokens)
        )
        self.selection_max_output_tokens = int(p.get("selection_max_output_tokens", self.max_output_tokens))
        self.max_total_ai_calls = int(p.get("max_total_ai_calls", p.get("max_ai_calls", 0)))
        self.max_ai_regime_calls = int(p.get("max_ai_regime_calls", p.get("max_calls", 0)))
        self.max_ai_selection_calls = int(p.get("max_ai_selection_calls", p.get("max_calls", 0)))
        self.ai_regime_interval = max(int(p.get("ai_regime_interval", 21)), 1)
        self.ai_regime_on_event = bool(p.get("ai_regime_on_event", True))
        self.ai_enabled = bool(p.get("ai_enabled", True))
        self.ai_selection_interval = max(int(p.get("ai_selection_interval", self.ai_regime_interval)), 1)
        self.candidate_top_n = max(int(p.get("candidate_top_n", 12)), 1)
        self.switch_min_hold_days = max(int(p.get("switch_min_hold_days", 30)), 0)
        raw_sensitiveness = p.get("sensitiveness")
        sensitivity_text = str(raw_sensitiveness).strip().lower() if raw_sensitiveness is not None else ""
        self.sensitivity_enabled = (
            raw_sensitiveness not in (None, "", False)
            and sensitivity_text not in {"off", "none", "false", "disabled"}
        )
        self.sensitiveness = self._normalize_sensitiveness(raw_sensitiveness)
        self.always_call_first = bool(p.get("always_call_first", True))
        self.active_pair_protection = bool(p.get("active_pair_protection", True))
        self.fail_open = bool(p.get("fail_open", True))
        self.retry_ai_after_failure = bool(p.get("retry_ai_after_failure", False))
        self.raw_response_debug_dir = str(
            p.get("raw_response_debug_dir", "results/evaluation/ai_regime_raw_responses")
        )
        self.precomputed_ai_regime_path = str(p.get("precomputed_ai_regime_path", ""))
        self.precomputed_ai_regime_only = bool(
            p.get("precomputed_ai_regime_only", bool(self.precomputed_ai_regime_path))
        )
        self.disable_regime_context = bool(p.get("disable_regime_context", False))
        self.disable_recent_rank_context = bool(p.get("disable_recent_rank_context", False))
        self.disable_benchmark_context = bool(p.get("disable_benchmark_context", False))
        self.disable_deterministic_baseline_context = bool(
            p.get("disable_deterministic_baseline_context", False)
        )
        self.deterministic_switch_decision = bool(p.get("deterministic_switch_decision", False))
        self.deterministic_pair_selection = bool(p.get("deterministic_pair_selection", False))

        self._step = 0
        self._ai_regime_call_count = 0
        self._ai_selection_call_count = 0
        self._last_ai_regime_step: int | None = None
        self._last_ai_selection_step: int | None = None
        self._last_ai_regime: dict[str, Any] | None = None
        self._last_ai_selection: dict[str, Any] | None = None
        self._last_raw_response: dict[str, Any] | None = None
        self._switch_history: list[int] = []
        self._last_global_regime: str | None = None
        self._precomputed_ai_regimes: dict[str, dict[str, Any]] = {}
        self._load_pair_regime_metrics(p.get("regime_suitability_path"))
        self._load_precomputed_ai_regimes(self.precomputed_ai_regime_path)

    def _load_precomputed_ai_regimes(self, path_value: Any) -> None:
        self._precomputed_ai_regimes = {}
        if not path_value:
            return
        path = Path(str(path_value))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return

        import csv

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_value = row.get("date") or row.get("timestamp")
                if not date_value:
                    continue
                regime = self._canonical_regime(row.get("ai_vol_regime", row.get("vol_regime", "middle")))
                if regime == "mid":
                    regime = "middle"
                if regime not in ("low", "middle", "high"):
                    continue
                key = str(date_value)[:10]
                self._precomputed_ai_regimes[key] = {
                    "vol_regime": regime,
                    "confidence": self._safe_float(row.get("confidence"), 0.0),
                    "reason": str(row.get("reason", ""))[:240],
                    "source": "precomputed",
                    "prediction_date": str(row.get("prediction_date", key))[:10],
                    "feature_end_date": str(row.get("feature_end_date", row.get("prediction_date", key)))[:10],
                    "error": str(row.get("error", "")),
                }

    def _load_pair_regime_metrics(self, path_value: Any) -> None:
        self._pair_regime_metrics: dict[str, list[dict[str, Any]]] = {}
        if not path_value:
            return
        path = Path(str(path_value))
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.is_dir():
            path = path / "pair_regime_suitability.csv"
        if not path.exists():
            return

        import csv

        valid_names = {pair.name for pair in self.pairs}
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair = str(row.get("pair", ""))
                if pair not in valid_names:
                    continue
                regime = self._canonical_regime(row.get("regime", "middle"))
                if regime == "mid":
                    regime = "middle"
                if regime not in ("low", "middle", "high"):
                    continue
                clean = {
                    "pair": pair,
                    "regime": regime,
                    "sharpe": self._safe_float(row.get("sharpe"), 0.0),
                    "max_drawdown": self._safe_float(row.get("max_drawdown"), 0.0),
                    "avg_turnover": self._safe_float(row.get("avg_turnover"), 0.0),
                    "realized_vol": self._safe_float(row.get("realized_vol"), 0.0),
                    "vol_tracking_error": self._safe_float(row.get("vol_tracking_error"), 0.0),
                    "n_days": int(self._safe_float(row.get("n_days"), 0.0)),
                }
                self._pair_regime_metrics.setdefault(regime, []).append(clean)

    def _default_endpoint(self) -> str:
        if self.provider == "nvidia":
            return self.nvidia_endpoint
        if self.api_format in ("chat", "chat_completions", "chat-completions"):
            return self.chat_completions_endpoint
        return self.responses_endpoint

    @staticmethod
    def _normalize_sensitiveness(value: Any) -> str:
        text = str(value or "normal").strip().lower().replace("-", "_")
        aliases = {
            "veryhigh": "very_high",
            "very_high": "very_high",
            "max": "very_high",
            "conservative": "low",
            "slow": "low",
            "stable": "low",
            "verylow": "very_low",
            "very_low": "very_low",
            "minimum": "very_low",
            "min": "very_low",
            "medium": "normal",
            "mid": "normal",
            "balanced": "normal",
            "default": "normal",
            "eager": "high",
            "aggressive": "high",
            "exploratory": "high",
        }
        text = aliases.get(text, text)
        return text if text in {"very_low", "low", "normal", "high", "very_high"} else "normal"

    def _sensitiveness_context(self) -> dict[str, str]:
        guidance = {
            "very_low": (
                "Very conservative: hold at least 90 trading days unless a broad group beats active across 20/60/100-day ranks."
            ),
            "low": (
                "Conservative: hold at least 60 trading days unless a group beats active across multiple time-scale ranks."
            ),
            "normal": (
                "Balanced: hold at least 30 trading days unless a group beats active across multiple time-scale ranks."
            ),
            "high": (
                "Eager: switch when one non-active pair has better short-term performance than the active pair."
            ),
            "very_high": (
                "Very eager: allow consecutive checkpoint switches when any non-active pair has better 20-day performance."
            ),
        }
        return {
            "level": self.sensitiveness,
            "guidance": guidance[self.sensitiveness],
        }

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match is None:
                action_match = re.search(r'"action"\s*:\s*"(hold|stay|switch|review|change)"', text, flags=re.I)
                if not action_match:
                    action_match = re.search(
                        r'"action"\s*:\s*"(hold|stay|switch|review|change)\b',
                        text,
                        flags=re.I,
                    )
                if action_match:
                    recovered: dict[str, Any] = {
                        "action": action_match.group(1).lower(),
                        "confidence": 0.0,
                        "recovered_from_truncated_json": True,
                    }
                    pair_match = re.search(r'"pair"\s*:\s*"([^"]+)"', text, flags=re.I)
                    if pair_match:
                        recovered["pair"] = pair_match.group(1)
                    return recovered
                lowered = text.lower()
                switch_phrases = (
                    "action switch",
                    "request switch",
                    "request a switch",
                    "should switch",
                    "switch review",
                    "choose switch",
                )
                hold_phrases = (
                    "action hold",
                    "should hold",
                    "choose hold",
                    "recommend hold",
                    "hold the active",
                    "hold active",
                )
                if any(phrase in lowered for phrase in switch_phrases):
                    return {
                        "action": "switch",
                        "confidence": 0.0,
                        "recovered_from_reasoning_text": True,
                    }
                if any(phrase in lowered for phrase in hold_phrases):
                    return {
                        "action": "hold",
                        "confidence": 0.0,
                        "recovered_from_reasoning_text": True,
                    }
                raise ValueError(f"AI response was not JSON: {text[:200]!r}")
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                chunk = match.group(0)
                action_match = re.search(
                    r'"action"\s*:\s*"(hold|stay|switch|review|change)"',
                    chunk,
                    flags=re.I,
                )
                if action_match:
                    recovered = {
                        "action": action_match.group(1).lower(),
                        "confidence": 0.0,
                        "recovered_from_truncated_json": True,
                    }
                    pair_match = re.search(r'"pair"\s*:\s*"([^"]+)"', chunk, flags=re.I)
                    if pair_match:
                        recovered["pair"] = pair_match.group(1)
                    return recovered
                raise
        if not isinstance(parsed, dict):
            raise ValueError("AI response JSON must be an object.")
        return parsed

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            text = choices[0].get("text")
            if isinstance(text, str) and text.strip():
                return text
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str) and content.strip():
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
            for key in ("reasoning_content", "reasoning"):
                text = message.get(key)
                if isinstance(text, str) and text.strip():
                    return text

        chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()

    @classmethod
    def _find_action_value(cls, value: Any) -> str:
        if isinstance(value, dict):
            action = str(value.get("action", "")).lower()
            if action in ("hold", "stay", "switch", "review", "change"):
                return action
            for item in value.values():
                found = cls._find_action_value(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = cls._find_action_value(item)
                if found:
                    return found
        elif isinstance(value, str):
            match = re.search(r'"action"\s*:\s*"(hold|stay|switch|review|change)"', value, flags=re.I)
            if match:
                return match.group(1).lower()
            text = value.strip().lower()
            if text in ("hold", "stay", "switch", "review", "change"):
                return text
        return ""

    @classmethod
    def _find_pair_value(cls, value: Any, valid_names: set[str]) -> str:
        if isinstance(value, dict):
            pair = str(value.get("pair", ""))
            if pair in valid_names:
                return pair
            for item in value.values():
                found = cls._find_pair_value(item, valid_names)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = cls._find_pair_value(item, valid_names)
                if found:
                    return found
        elif isinstance(value, str):
            for name in sorted(valid_names, key=len, reverse=True):
                if name in value:
                    return name
        return ""

    def _save_raw_response(self, error: str, call_kind: str = "ai") -> str:
        if self._last_raw_response is None:
            return ""
        try:
            out_dir = Path(self.raw_response_debug_dir)
            if not out_dir.is_absolute():
                out_dir = Path.cwd() / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            call_count = (
                self._ai_regime_call_count
                if call_kind == "regime"
                else self._ai_selection_call_count
            )
            path = out_dir / f"ai_{call_kind}_step_{self._step:06d}_call_{call_count:04d}.json"
            path.write_text(
                json.dumps(
                    {
                        "step": self._step,
                        "call_kind": call_kind,
                        "call_count": call_count,
                        "model": self.model,
                        "provider": self.provider,
                        "endpoint": self.endpoint,
                        "error": error,
                        "raw_response": self._last_raw_response,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            return str(path)
        except Exception:
            return ""

    def _call_ai(
        self,
        prompt: str,
        system_prompt: str,
        call_kind: str,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("AI API key is empty.")
        if self.max_total_ai_calls > 0 and (
            self._ai_regime_call_count + self._ai_selection_call_count
        ) >= self.max_total_ai_calls:
            raise RuntimeError("AI total max call limit reached.")
        if call_kind == "regime":
            if self.max_ai_regime_calls > 0 and self._ai_regime_call_count >= self.max_ai_regime_calls:
                raise RuntimeError("AI regime max call limit reached.")
            self._ai_regime_call_count += 1
        else:
            if self.max_ai_selection_calls > 0 and self._ai_selection_call_count >= self.max_ai_selection_calls:
                raise RuntimeError("AI selection max call limit reached.")
            self._ai_selection_call_count += 1

        if self.request_min_interval_seconds > 0 and self._last_ai_request_monotonic > 0:
            elapsed = time.monotonic() - self._last_ai_request_monotonic
            wait_seconds = self.request_min_interval_seconds - elapsed
            if wait_seconds > 0:
                time.sleep(wait_seconds)

        token_limit = int(max_output_tokens if max_output_tokens is not None else self.max_output_tokens)
        if self.api_format in ("chat", "chat_completions", "chat-completions") or self.provider == "nvidia":
            request_payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            if token_limit > 0:
                request_payload["max_tokens"] = token_limit
            if self.temperature is not None:
                request_payload["temperature"] = self.temperature
            if self.response_format:
                request_payload["response_format"] = self.response_format
            if self.reasoning_effort:
                request_payload["reasoning_effort"] = self.reasoning_effort
        else:
            request_payload = {
                "model": self.model,
                "instructions": system_prompt,
                "input": prompt,
            }
            if token_limit > 0:
                request_payload["max_output_tokens"] = token_limit
            if self.temperature is not None:
                request_payload["temperature"] = self.temperature
            if self.reasoning_effort:
                request_payload["reasoning"] = {"effort": self.reasoning_effort}

        context = mp.get_context("fork")
        queue: Any = context.Queue(maxsize=1)
        process = context.Process(
            target=_post_json_worker,
            args=(self.endpoint, request_payload, self.api_key, self.timeout, queue),
        )
        self._last_ai_request_monotonic = time.monotonic()
        process.start()
        process.join(self.timeout if self.timeout > 0 else None)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            raise RuntimeError(f"AI {call_kind} request exceeded {self.timeout:.1f}s")
        if queue.empty():
            raise RuntimeError(f"AI {call_kind} request failed without provider response")
        status, data = queue.get()
        if status != "ok":
            raise RuntimeError(f"AI {call_kind} request failed: {data}")

        self._last_raw_response = data
        try:
            return self._parse_json(self._extract_text(data))
        except Exception as exc:
            debug_path = self._save_raw_response(str(exc), call_kind=call_kind)
            if debug_path:
                raise ValueError(f"{exc}; raw_response_saved={debug_path}") from exc
            raise

    def _call_ai_regime(self, prompt: str) -> dict[str, Any]:
        return self._call_ai(prompt, self.system_prompt, "regime")

    def _call_ai_selection(self, prompt: str) -> dict[str, Any]:
        return self._call_ai(
            prompt,
            self.selection_system_prompt,
            "selection",
            max_output_tokens=self.selection_max_output_tokens,
        )

    def _call_ai_switch_decision(self, prompt: str) -> dict[str, Any]:
        return self._call_ai(
            prompt,
            self.switch_system_prompt,
            "selection",
            max_output_tokens=self.switch_decision_max_output_tokens,
        )

    def _compact_performance_context(self, performance_metrics: dict[str, Any]) -> dict[str, Any]:
        active_name = self._active_pair.name if self._active_pair is not None else ""
        active_perf = performance_metrics.get(active_name)
        benchmark = performance_metrics.get("benchmark")
        out: dict[str, Any] = {
            key: round(self._safe_float(performance_metrics.get(key)), 6)
            for key in ("rolling_sharpe", "drawdown", "realized_vol", "vol_tracking_error")
            if key in performance_metrics
        }
        if isinstance(active_perf, dict):
            out["active_pair"] = active_perf
        if isinstance(benchmark, dict) and not self.disable_benchmark_context:
            out["benchmark"] = benchmark
        return out

    def _compact_pair_performance(
        self,
        pair_name: str,
        performance_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        pair_perf = performance_metrics.get(pair_name)
        if not isinstance(pair_perf, dict):
            return {}
        keep: dict[str, Any] = {}
        for key in (
            "obs",
            "rolling_sharpe",
            "drawdown",
            "realized_vol",
            "vol_tracking_error",
            "turnover",
            "cvar_95",
            "tail_loss",
        ):
            if key in pair_perf:
                keep[key] = round(self._safe_float(pair_perf.get(key)), 6)
        for key in ("trailing_10d", "trailing_63d", "trailing_126d"):
            value = pair_perf.get(key)
            if isinstance(value, dict):
                keep[key] = {
                    sub_key: round(self._safe_float(value.get(sub_key)), 6)
                    for sub_key in (
                        "trailing_return",
                        "annualized_return",
                        "rolling_sharpe",
                        "drawdown",
                        "realized_vol",
                        "vol_band_error",
                        "turnover",
                        "cvar_95",
                        "tail_loss",
                    )
                    if sub_key in value
                }
        return keep

    def _recent_rank_context(
        self,
        performance_metrics: dict[str, Any],
        pair_names: set[str] | None = None,
        top_n: int = 12,
    ) -> dict[str, Any]:
        if self.disable_recent_rank_context:
            return {}
        raw = performance_metrics.get("recent_pair_rankings", {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, Any] = {}
        for window, block in raw.items():
            if not isinstance(block, dict):
                continue
            rows = block.get("ranks", [])
            if not isinstance(rows, list):
                rows = []
            compact_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                pair = str(row.get("pair", ""))
                if not pair:
                    continue
                if pair_names is None:
                    include = len(compact_rows) < top_n
                else:
                    include = pair in pair_names or len(compact_rows) < top_n
                if not include:
                    continue
                compact_rows.append(
                    {
                        "rank": int(self._safe_float(row.get("rank"), 0.0)),
                        "pair": pair,
                        "obs": int(self._safe_float(row.get("obs"), 0.0)),
                        "turnover": round(self._safe_float(row.get("turnover"), 0.0), 6),
                        "drawdown": round(self._safe_float(row.get("drawdown"), 0.0), 6),
                        "sharpe": round(self._safe_float(row.get("sharpe"), 0.0), 6),
                        "cvar_95": round(self._safe_float(row.get("cvar_95"), 0.0), 6),
                        "realized_vol": round(self._safe_float(row.get("realized_vol"), 0.0), 6),
                        "vol_tracking_error": round(self._safe_float(row.get("vol_tracking_error"), 0.0), 6),
                        "annualized_return": round(self._safe_float(row.get("annualized_return"), 0.0), 6),
                    }
                )
            out[str(window)] = {
                "window_days": int(self._safe_float(block.get("window_days"), 0.0)),
                "rank_rule": str(block.get("rank_rule", "")),
                "ranks": compact_rows[: max(top_n, len(pair_names or []))],
            }
        return out

    def _recent_rank_for_pair(
        self,
        performance_metrics: dict[str, Any],
        pair_name: str | None,
    ) -> dict[str, Any]:
        if not pair_name:
            return {}
        context = self._recent_rank_context(performance_metrics, {pair_name}, top_n=0)
        out: dict[str, Any] = {}
        for window, block in context.items():
            for row in block.get("ranks", []):
                if str(row.get("pair")) == str(pair_name):
                    out[window] = row
                    break
        return out

    def _recent_rank_rows_by_window(
        self,
        performance_metrics: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        context = self._recent_rank_context(performance_metrics, None, top_n=10**6)
        out: dict[str, list[dict[str, Any]]] = {}
        for window, block in context.items():
            rows = block.get("ranks", []) if isinstance(block, dict) else []
            out[str(window)] = [row for row in rows if isinstance(row, dict)]
        return out

    def _sensitivity_switch_context(
        self,
        performance_metrics: dict[str, Any],
        candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        active_name = self._active_pair.name if self._active_pair is not None else ""
        valid_names = {str(row.get("name")) for row in (candidates or []) if row.get("name")}
        if not valid_names:
            valid_names = {pair.name for pair in self.pairs}
        valid_names.discard(active_name)
        windows = self._recent_rank_rows_by_window(performance_metrics)

        active_by_window: dict[str, dict[str, Any]] = {}
        rows_by_pair: dict[str, dict[str, dict[str, Any]]] = {}
        for window, rows in windows.items():
            for row in rows:
                pair = str(row.get("pair", ""))
                if not pair:
                    continue
                if pair == active_name:
                    active_by_window[window] = row
                if pair in valid_names:
                    rows_by_pair.setdefault(pair, {})[window] = row

        short_window = "20d" if "20d" in windows else (sorted(windows)[0] if windows else "")
        active_short = active_by_window.get(short_window, {})
        active_short_sharpe = self._safe_float(active_short.get("sharpe"), 0.0)
        active_short_return = self._safe_float(active_short.get("annualized_return"), 0.0)

        short_term_challengers = []
        multi_scale_challengers = []
        for pair, by_window in rows_by_pair.items():
            short = by_window.get(short_window, {})
            short_sharpe = self._safe_float(short.get("sharpe"), 0.0)
            short_return = self._safe_float(short.get("annualized_return"), 0.0)
            short_beats_active = bool(
                short
                and (
                    short_sharpe > active_short_sharpe
                    or (
                        short_sharpe >= active_short_sharpe
                        and short_return > active_short_return
                    )
                )
            )
            beats_windows = []
            for window in ("20d", "60d", "100d"):
                row = by_window.get(window)
                active_row = active_by_window.get(window)
                if not row or not active_row:
                    continue
                rank = int(self._safe_float(row.get("rank"), 10**9))
                active_rank = int(self._safe_float(active_row.get("rank"), 10**9))
                sharpe = self._safe_float(row.get("sharpe"), 0.0)
                active_sharpe = self._safe_float(active_row.get("sharpe"), 0.0)
                drawdown = abs(self._safe_float(row.get("drawdown"), 0.0))
                active_drawdown = abs(self._safe_float(active_row.get("drawdown"), 0.0))
                if rank < active_rank and (sharpe >= active_sharpe or drawdown <= active_drawdown):
                    beats_windows.append(window)
            row_out = {
                "pair": pair,
                "short_window": short_window,
                "short_sharpe": round(short_sharpe, 6),
                "active_short_sharpe": round(active_short_sharpe, 6),
                "short_annualized_return": round(short_return, 6),
                "active_short_annualized_return": round(active_short_return, 6),
                "short_term_beats_active": short_beats_active,
                "beats_windows": beats_windows,
                "beats_window_count": len(beats_windows),
            }
            if short_beats_active:
                short_term_challengers.append(row_out)
            if len(beats_windows) >= 2:
                multi_scale_challengers.append(row_out)

        short_term_challengers.sort(
            key=lambda row: (
                -float(row["short_sharpe"]),
                -float(row["short_annualized_return"]),
                str(row["pair"]),
            )
        )
        multi_scale_challengers.sort(
            key=lambda row: (
                -int(row["beats_window_count"]),
                -float(row["short_sharpe"]),
                str(row["pair"]),
            )
        )
        required_group = {"normal": 2, "low": 3, "very_low": 4}.get(self.sensitiveness, 1)
        min_hold_days = {"very_high": 0, "high": 0, "normal": 30, "low": 60, "very_low": 90}[
            self.sensitiveness
        ]
        hold_days = int(self._active_since)
        group_exception = len(multi_scale_challengers) >= required_group
        return {
            "sensitiveness": self.sensitiveness,
            "active_pair": active_name,
            "hold_days": hold_days,
            "minimum_hold_days": min_hold_days,
            "minimum_hold_satisfied": hold_days >= min_hold_days,
            "required_multi_scale_group_size": required_group,
            "group_exception": group_exception,
            "short_term_challenger_count": len(short_term_challengers),
            "multi_scale_challenger_count": len(multi_scale_challengers),
            "top_short_term_challengers": short_term_challengers[:5],
            "top_multi_scale_challengers": multi_scale_challengers[:5],
        }

    def _ai_regime_payload(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> str:
        payload = {
            "instruction": (
                "Classify the current volatility regime for router scoring. "
                "Use low/middle/high only. Do not choose a strategy pair."
            ),
            "current_rule_regime": market_features.get("vol_regime"),
            "rolling_market_features": {
                key: market_features.get(key)
                for key in (
                    "rolling_vol",
                    "rolling_mean",
                    "rolling_skew",
                    "window_obs",
                    "vol_regime",
                    "intraday_realized_vol",
                )
                if key in market_features
            },
            "recent_router_performance": self._compact_performance_context(performance_metrics),
            "previous_ai_regime": self._last_ai_regime,
        }
        return json.dumps(payload, default=str)

    def _regime_event_reason(self, market_features: dict[str, Any]) -> str | None:
        if self._last_ai_regime is None:
            return "first_regime"
        return None

    def _should_call_ai_regime(self, market_features: dict[str, Any]) -> tuple[bool, str]:
        if not self.ai_enabled:
            return False, "ai_disabled"
        if self.max_ai_regime_calls > 0 and self._ai_regime_call_count >= self.max_ai_regime_calls:
            return False, "max_ai_regime_calls"
        event_reason = self._regime_event_reason(market_features)
        if event_reason:
            return True, event_reason
        if self._last_ai_regime_step is None:
            return True, "first_regime"
        gap = self._step - self._last_ai_regime_step
        if gap >= self.ai_regime_interval:
            return True, f"interval_{self.ai_regime_interval}"
        return False, f"hold_ai_regime_{gap}"

    def _with_ai_regime(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
        timestamp: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        enriched = dict(market_features)
        if self.disable_regime_context:
            enriched["rule_vol_regime"] = market_features.get("vol_regime")
            enriched["vol_regime"] = "middle"
            enriched["ai_regime"] = {}
            enriched["ai_regime_call_count"] = self._ai_regime_call_count
            enriched["ai_regime_error"] = ""
            return enriched, {}, "regime_context_disabled"
        regime_key = str(timestamp)[:10] if timestamp is not None else ""
        if regime_key and regime_key in self._precomputed_ai_regimes:
            precomputed = dict(self._precomputed_ai_regimes[regime_key])
            prediction_date = str(precomputed.get("prediction_date", regime_key))[:10]
            feature_end_date = str(precomputed.get("feature_end_date", prediction_date))[:10]
            no_lookahead = prediction_date <= regime_key and feature_end_date < regime_key
            if no_lookahead:
                self._last_ai_regime = precomputed
                self._last_ai_regime["call_reason"] = "precomputed_ai_regime"
                self._last_ai_regime_step = self._step
                enriched["rule_vol_regime"] = market_features.get("vol_regime")
                enriched["vol_regime"] = self._last_ai_regime["vol_regime"]
                enriched["ai_regime"] = dict(self._last_ai_regime)
                enriched["ai_regime_call_count"] = self._ai_regime_call_count
                enriched["ai_regime_error"] = self._last_ai_regime.get("error", "")
                return enriched, dict(self._last_ai_regime), "precomputed_ai_regime"
        if self.precomputed_ai_regime_only and self._precomputed_ai_regimes:
            fallback_regime = (
                self._last_ai_regime.get("vol_regime")
                if self._last_ai_regime is not None
                else "middle"
            )
            self._last_ai_regime = {
                "vol_regime": fallback_regime,
                "confidence": 0.0,
                "reason": f"Missing precomputed AI regime for {regime_key}; held previous regime",
                "call_reason": "missing_precomputed_ai_regime",
                "source": "precomputed_missing",
                "error": "",
            }
            enriched["rule_vol_regime"] = market_features.get("vol_regime")
            enriched["vol_regime"] = fallback_regime
            enriched["ai_regime"] = dict(self._last_ai_regime)
            enriched["ai_regime_call_count"] = self._ai_regime_call_count
            enriched["ai_regime_error"] = ""
            return enriched, dict(self._last_ai_regime), "missing_precomputed_ai_regime"

        should_call, reason = self._should_call_ai_regime(enriched)
        ai_error = ""

        if should_call:
            try:
                response = self._call_ai_regime(self._ai_regime_payload(enriched, performance_metrics))
                regime = self._canonical_regime(response.get("vol_regime", "middle"))
                if regime == "mid":
                    regime = "middle"
                if regime not in ("low", "middle", "high"):
                    raise ValueError(f"AI regime must be low/middle/high, got {regime!r}")
                self._last_ai_regime = {
                    "vol_regime": regime,
                    "confidence": self._safe_float(response.get("confidence"), 0.0),
                    "reason": str(response.get("reason", ""))[:240],
                    "call_reason": reason,
                    "error": "",
                }
                self._last_ai_regime_step = self._step
            except Exception as exc:
                ai_error = str(exc)
                fallback_source = (
                    self._last_ai_regime.get("vol_regime")
                    if self._last_ai_regime is not None
                    else "middle"
                )
                fallback_regime = self._canonical_regime(fallback_source)
                if fallback_regime == "mid":
                    fallback_regime = "middle"
                if fallback_regime not in ("low", "middle", "high"):
                    fallback_regime = "middle"
                self._last_ai_regime = {
                    "vol_regime": fallback_regime,
                    "confidence": 0.0,
                    "reason": "AI regime failed; held previous AI regime until retry",
                    "call_reason": reason,
                    "error": ai_error,
                }
                if not self.retry_ai_after_failure:
                    self._last_ai_regime_step = self._step
                if not self.fail_open:
                    raise

        if self._last_ai_regime is not None:
            enriched["rule_vol_regime"] = market_features.get("vol_regime")
            enriched["vol_regime"] = self._last_ai_regime["vol_regime"]
            enriched["ai_regime"] = dict(self._last_ai_regime)
            enriched["ai_regime_call_count"] = self._ai_regime_call_count
            enriched["ai_regime_error"] = ai_error
        else:
            enriched["ai_regime_error"] = ai_error
        return enriched, dict(self._last_ai_regime or {}), reason

    def _regime_pair_pool(
        self,
        regime: str,
        performance_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.disable_regime_context:
            out = []
            for pair in self.pairs:
                pair_name = pair.name
                perf = performance_metrics.get(pair_name, {})
                if not isinstance(perf, dict):
                    perf = {}
                out.append(
                    {
                        "name": pair_name,
                        "recent_performance": self._compact_pair_performance(
                            pair_name, performance_metrics
                        ),
                        "overall_rank_context": {
                            "turnover": round(self._safe_float(perf.get("turnover"), 0.0), 6),
                            "drawdown": round(self._safe_float(perf.get("drawdown"), 0.0), 6),
                            "sharpe": round(self._safe_float(perf.get("rolling_sharpe"), 0.0), 6),
                            "realized_vol": round(self._safe_float(perf.get("realized_vol"), 0.0), 6),
                            "vol_tracking_error": round(
                                self._safe_float(perf.get("vol_tracking_error"), 0.0), 6
                            ),
                        },
                        "vs_rv22_naive_scaling": self._pair_vs_benchmark(
                            pair_name, performance_metrics
                        ),
                    }
                )
            return out
        regime = self._canonical_regime(regime)
        if regime == "mid":
            regime = "middle"

        valid_pairs = {pair.name for pair in self.pairs}
        rows = self._rolling_regime_rows(regime, performance_metrics, valid_pairs)
        if not rows:
            rows = [
                dict(row)
                for row in self._pair_regime_metrics.get(regime, [])
                if str(row.get("pair")) in valid_pairs
            ]
        if not rows:
            rows = []
            for pair in self.pairs:
                perf = performance_metrics.get(pair.name, {})
                if not isinstance(perf, dict):
                    perf = {}
                rows.append(
                    {
                        "pair": pair.name,
                        "regime": regime,
                        "avg_turnover": self._safe_float(perf.get("turnover"), 0.0),
                        "max_drawdown": self._safe_float(perf.get("drawdown"), 0.0),
                        "sharpe": self._safe_float(perf.get("rolling_sharpe"), 0.0),
                        "realized_vol": self._safe_float(perf.get("realized_vol"), 0.0),
                        "vol_tracking_error": self._safe_float(perf.get("vol_tracking_error"), 0.0),
                        "n_days": int(self._safe_float(perf.get("obs"), 0.0)),
                    }
                )

        out = []
        seen: set[str] = set()
        for row in rows:
            pair_name = str(row["pair"])
            if pair_name in seen:
                continue
            seen.add(pair_name)
            out.append(
                {
                    "name": pair_name,
                    "regime_history": {
                        "regime": regime,
                        "avg_turnover": round(self._safe_float(row.get("avg_turnover"), 0.0), 6),
                        "max_drawdown": round(self._safe_float(row.get("max_drawdown"), 0.0), 6),
                        "sharpe": round(self._safe_float(row.get("sharpe"), 0.0), 6),
                        "realized_vol": round(self._safe_float(row.get("realized_vol"), 0.0), 6),
                        "vol_tracking_error": round(self._safe_float(row.get("vol_tracking_error"), 0.0), 6),
                        "n_days": int(self._safe_float(row.get("n_days"), 0.0)),
                    },
                    "recent_performance": self._compact_pair_performance(pair_name, performance_metrics),
                    "vs_rv22_naive_scaling": self._pair_vs_benchmark(pair_name, performance_metrics),
                }
            )
        self._attach_all_regime_ranks(out, performance_metrics)
        return out

    def _attach_all_regime_ranks(
        self,
        candidates: list[dict[str, Any]],
        performance_metrics: dict[str, Any],
    ) -> None:
        valid_names = {str(row["name"]) for row in candidates}
        by_name = {str(row["name"]): row for row in candidates}
        rank_table: dict[str, dict[str, dict[str, Any]]] = {
            name: {} for name in valid_names
        }
        for regime in ("low", "middle", "high"):
            rows = self._rolling_regime_rows(regime, performance_metrics, valid_names)
            if not rows:
                rows = [
                    dict(row)
                    for row in self._pair_regime_metrics.get(regime, [])
                    if str(row.get("pair")) in valid_names
                ]
            if not rows:
                rows = [
                    {
                        "pair": name,
                        "regime": regime,
                        "avg_turnover": self._safe_float(
                            by_name[name].get("recent_performance", {}).get("turnover"), 0.0
                        ),
                        "max_drawdown": self._safe_float(
                            by_name[name].get("recent_performance", {}).get("drawdown"), 0.0
                        ),
                        "sharpe": self._safe_float(
                            by_name[name].get("recent_performance", {}).get("rolling_sharpe"), 0.0
                        ),
                    }
                    for name in valid_names
                ]

            selection_ranks = {
                str(row["pair"]): rank
                for rank, row in enumerate(
                    self._sort_rows_by_turnover_drawdown_sharpe(rows),
                    start=1,
                )
            }

            for row in rows:
                name = str(row["pair"])
                if name not in valid_names:
                    continue
                rank_table[name][regime] = {
                    "selection_rank": selection_ranks.get(name, 0),
                    "avg_turnover": round(self._safe_float(row.get("avg_turnover"), 0.0), 6),
                    "max_drawdown": round(self._safe_float(row.get("max_drawdown"), 0.0), 6),
                    "sharpe": round(self._safe_float(row.get("sharpe"), 0.0), 6),
                    "rank_rule": "turnover_bucket_then_drawdown_then_sharpe",
                }

        active_regime = candidates[0].get("regime_history", {}).get("regime", "middle") if candidates else "middle"
        for row in candidates:
            name = str(row["name"])
            all_regime_ranks = rank_table.get(name, {})
            row["all_regime_ranks"] = all_regime_ranks
            row["active_regime_ranks"] = all_regime_ranks.get(str(active_regime), {})
            row["rank_priority_order"] = "use_ai_regime_then_selection_rank"

    def _sort_rows_by_turnover_drawdown_sharpe(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        turnovers = sorted(self._safe_float(row.get("avg_turnover"), 0.0) for row in rows)
        gaps = [b - a for a, b in zip(turnovers, turnovers[1:]) if b > a]
        tolerance = max((sum(gaps) / len(gaps)) if gaps else 0.0, 0.0025)
        sorted_by_turnover = sorted(rows, key=lambda row: self._safe_float(row.get("avg_turnover"), 0.0))

        buckets: list[list[dict[str, Any]]] = []
        for row in sorted_by_turnover:
            turnover = self._safe_float(row.get("avg_turnover"), 0.0)
            if not buckets:
                buckets.append([row])
                continue
            bucket_anchor = self._safe_float(buckets[-1][0].get("avg_turnover"), 0.0)
            if turnover - bucket_anchor <= tolerance:
                buckets[-1].append(row)
            else:
                buckets.append([row])

        out: list[dict[str, Any]] = []
        for bucket in buckets:
            out.extend(
                sorted(
                    bucket,
                    key=lambda row: (
                        self._safe_float(row.get("max_drawdown"), 0.0),
                        -self._safe_float(row.get("sharpe"), 0.0),
                        self._safe_float(row.get("avg_turnover"), 0.0),
                    ),
                )
            )
        return out

    def _pair_vs_benchmark(self, pair_name: str, performance_metrics: dict[str, Any]) -> dict[str, float]:
        if self.disable_benchmark_context:
            return {}
        pair_perf = performance_metrics.get(pair_name)
        benchmark = performance_metrics.get("benchmark")
        if not isinstance(pair_perf, dict) or not isinstance(benchmark, dict):
            return {}
        out: dict[str, float] = {}
        comparisons = {
            "rolling_sharpe": 1.0,
            "drawdown": -1.0,
            "realized_vol": -1.0,
            "vol_tracking_error": -1.0,
            "turnover": -1.0,
        }
        for key, direction in comparisons.items():
            if key in pair_perf and key in benchmark:
                diff = self._safe_float(pair_perf.get(key), 0.0) - self._safe_float(benchmark.get(key), 0.0)
                out[f"{key}_minus_benchmark"] = round(diff, 6)
                out[f"{key}_better_than_benchmark"] = 1.0 if direction * diff > 0 else 0.0
        return out

    def _deterministic_baseline_context(
        self,
        performance_metrics: dict[str, Any],
        valid_names: set[str] | None = None,
    ) -> dict[str, Any]:
        if self.disable_deterministic_baseline_context:
            return {}
        raw_rows = performance_metrics.get("deterministic_pair_ranking", [])
        if not isinstance(raw_rows, list):
            raw_rows = []
        raw_best_pair = str(performance_metrics.get("deterministic_best_pair", self.default_pair.name))
        active_name = self._active_pair.name if self._active_pair is not None else ""

        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            pair_name = str(raw.get("pair", ""))
            if not pair_name or (valid_names is not None and pair_name not in valid_names):
                continue
            rows.append(
                {
                    "rank": int(self._safe_float(raw.get("rank"), 0.0)),
                    "pair": pair_name,
                    "turnover": round(self._safe_float(raw.get("turnover"), 0.0), 6),
                    "drawdown": round(self._safe_float(raw.get("drawdown"), 0.0), 6),
                    "sharpe": round(self._safe_float(raw.get("sharpe"), 0.0), 6),
                    "deterministic_score": round(
                        self._safe_float(raw.get("deterministic_score"), 0.0), 6
                    ),
                    "is_rank_best": bool(raw.get("is_rank_best", False)),
                    "is_train_score_best": bool(raw.get("is_train_score_best", pair_name == raw_best_pair)),
                    "is_active": pair_name == active_name,
                }
            )

        rows.sort(key=lambda row: row["rank"] if row["rank"] > 0 else 10**9)
        best_row = next((row for row in rows if row["is_train_score_best"]), None)
        rank_best_row = next((row for row in rows if row["is_rank_best"]), rows[0] if rows else None)
        best_pair = str(best_row.get("pair", "")) if isinstance(best_row, dict) else ""
        return {
            "rank_rule": performance_metrics.get(
                "deterministic_pair_rank_rule",
                "rank pairs by turnover ascending, then drawdown ascending, then Sharpe descending",
            ),
            "train_score_best_pair": best_pair,
            "train_score_best": best_row or {},
            "rank_best": rank_best_row or {},
            "active_is_train_score_best": active_name == best_pair,
            "ranked_pairs": rows,
        }

    def _rolling_regime_rows(
        self,
        regime: str,
        performance_metrics: dict[str, Any],
        valid_pairs: set[str],
    ) -> list[dict[str, Any]]:
        if self.disable_regime_context:
            return []
        raw = performance_metrics.get("regime_pair_history")
        if not isinstance(raw, dict):
            return []
        by_regime = raw.get(regime)
        if not isinstance(by_regime, dict):
            return []
        rows = []
        for pair_name, metrics in by_regime.items():
            if pair_name not in valid_pairs or not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "pair": pair_name,
                    "regime": regime,
                    "avg_turnover": self._safe_float(metrics.get("avg_turnover"), 0.0),
                    "max_drawdown": self._safe_float(metrics.get("max_drawdown"), 0.0),
                    "sharpe": self._safe_float(metrics.get("sharpe"), 0.0),
                    "realized_vol": self._safe_float(metrics.get("realized_vol"), 0.0),
                    "vol_tracking_error": self._safe_float(metrics.get("vol_tracking_error"), 0.0),
                    "n_days": int(self._safe_float(metrics.get("n_days"), 0.0)),
                }
            )
        return rows

    def _selection_payload(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> str:
        global_change_reason = self._global_regime_event_reason(market_features)
        valid_names = {str(row["name"]) for row in candidates}
        deterministic_baseline = self._deterministic_baseline_context(performance_metrics, valid_names)
        baseline_by_name = {
            str(row.get("pair")): row
            for row in deterministic_baseline.get("ranked_pairs", [])
            if isinstance(row, dict)
        }
        payload_candidates = []
        for row in candidates:
            enriched_row = dict(row)
            enriched_row["deterministic_baseline"] = baseline_by_name.get(str(row["name"]), {})
            enriched_row["recent_100_60_20_rank_context"] = self._recent_rank_for_pair(
                performance_metrics,
                str(row["name"]),
            )
            if self.disable_deterministic_baseline_context:
                enriched_row.pop("deterministic_baseline", None)
            payload_candidates.append(enriched_row)
        instruction_parts = [
            "Choose the best supplied pair for the next ten trading days after switch review approved switching. ",
            "The current active pair has been excluded from the supplied pair list; choose only from the supplied alternatives. ",
            "Return only {\"pair\":\"<supplied pair name>\"}.",
        ]
        if self.sensitivity_enabled:
            instruction_parts.extend(
                [
                    f"Sensitiveness is {self.sensitiveness}: {self._sensitiveness_context()['guidance']} ",
                    "For high/very_high sensitivity, prioritize the supplied pair with the strongest short-term rank/performance improvement. ",
                    "For normal/low/very_low sensitivity, prefer challengers that beat active across multiple time scales. ",
                ]
            )
        if not self.disable_regime_context:
            instruction_parts.extend(
                [
                    "The supplied pair list includes active-regime ranks plus 100/60/20-day ranks computed through the previous trading day. ",
                    "You must use each pair's active-regime selection_rank as the dominant decision input. ",
                    "Do not override active-regime selection_rank for small recent-performance differences. ",
                    "If AI regime is low use low selection_rank; if middle use middle selection_rank; if high use high selection_rank. ",
                    "The single selection_rank already sorts turnover first, drawdown among reasonable-turnover peers second, and Sharpe third.",
                ]
            )
        else:
            instruction_parts.append(
                "No regime labels, regime history, or regime-conditioned ranks are supplied in this run. "
            )
        if not self.disable_deterministic_baseline_context:
            instruction_parts.extend(
                [
                    " The deterministic_baseline list is ranked by train-window turnover, drawdown, then Sharpe and marks the train-score best pair.",
                    " Prefer the marked train-score best pair unless another candidate has a clearly stronger active-regime rank and benchmark tradeoff.",
                ]
            )
        if not self.disable_benchmark_context:
            instruction_parts.append(
                " Use RV22 + naive-scaling benchmark and recent performance only as tie-breakers or safety checks."
            )
        payload = {
            "instruction": "".join(instruction_parts),
            "deterministic_baseline": deterministic_baseline,
            "global_regime_monitor": {
                "current_global_regime": self._current_global_regime(market_features),
                "previous_global_regime": self._last_global_regime,
                "changed": global_change_reason is not None,
                "strong_switch_signal": global_change_reason is not None,
                "change_reason": global_change_reason or "",
            },
            "market_features": {
                key: market_features.get(key)
                for key in (
                    "rolling_vol",
                    "rule_vol_regime",
                    "vol_regime",
                    "intraday_realized_vol",
                )
                if key in market_features
            },
            "rv22_naive_scaling_benchmark": performance_metrics.get("benchmark", {}),
            "recent_pair_rankings_100_60_20": self._recent_rank_context(
                performance_metrics,
                valid_names,
                top_n=self.candidate_top_n,
            ),
            "pairs": payload_candidates,
        }
        if self.sensitivity_enabled:
            payload["sensitiveness"] = self._sensitiveness_context()
            payload["sensitivity_switch_context"] = self._sensitivity_switch_context(
                performance_metrics,
                candidates,
            )
        if not self.disable_regime_context:
            payload["predicted_regime"] = market_features.get("vol_regime")
            payload["ai_regime"] = market_features.get("ai_regime", {})
        if self.disable_deterministic_baseline_context:
            payload.pop("deterministic_baseline", None)
        if self.disable_benchmark_context:
            payload.pop("rv22_naive_scaling_benchmark", None)
        if self.disable_recent_rank_context:
            payload.pop("recent_pair_rankings_100_60_20", None)
        return json.dumps(payload, default=str)

    def _candidate_selection_rank(self, candidate: dict[str, Any]) -> tuple[int, str]:
        if self.disable_regime_context:
            recent = candidate.get("recent_performance") or {}
            overall = candidate.get("overall_rank_context") or {}
            turnover = self._safe_float(recent.get("turnover", overall.get("turnover")), 0.0)
            drawdown = self._safe_float(recent.get("drawdown", overall.get("drawdown")), 0.0)
            sharpe = self._safe_float(
                recent.get("rolling_sharpe", overall.get("sharpe")),
                0.0,
            )
            return int(turnover * 1_000_000), f"{drawdown:020.12f}:{-sharpe:020.12f}:{candidate.get('name', '')}"
        rank = self._safe_float(
            (candidate.get("active_regime_ranks") or {}).get("selection_rank"),
            0.0,
        )
        if rank <= 0:
            rank = 10**9
        return int(rank), str(candidate.get("name", ""))

    def _limit_selection_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(candidates) <= self.candidate_top_n:
            return candidates
        return sorted(candidates, key=self._candidate_selection_rank)[: self.candidate_top_n]

    def _switch_decision_payload(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> str:
        global_change_reason = self._global_regime_event_reason(market_features)
        active_name = self._active_pair.name if self._active_pair is not None else None
        deterministic_baseline = self._deterministic_baseline_context(performance_metrics)
        compact_baseline = {
            "rank_rule": deterministic_baseline.get("rank_rule"),
            "train_score_best_pair": deterministic_baseline.get("train_score_best_pair"),
            "train_score_best": deterministic_baseline.get("train_score_best"),
            "rank_best": deterministic_baseline.get("rank_best"),
            "active_is_train_score_best": deterministic_baseline.get("active_is_train_score_best"),
        }
        instruction = (
            "Return only {\"action\":\"hold\"} or {\"action\":\"switch\"}. "
            "Regime/context change or weak active recent rank favors switch. "
            "Strong active rank and low drawdown favors hold."
        )
        if self.sensitivity_enabled:
            instruction = (
                "Return only {\"action\":\"hold\"} or {\"action\":\"switch\"}. "
                "For high/very_high sensitivity, switch when one non-active pair has better short-term performance than active. "
                "For normal sensitivity, hold active for at least 30 trading days unless a group of pairs beats active across multiple time-scale ranks. "
                "For low sensitivity, hold active for at least 60 trading days unless a group of pairs beats active across multiple time-scale ranks. "
                "For very_low sensitivity, use a 90 trading-day hold gate and require an even broader multi-scale challenger group. "
                f"Sensitiveness is {self.sensitiveness}: {self._sensitiveness_context()['guidance']}"
            )
        payload = {
            "instruction": instruction,
            "active_pair": active_name,
            "deterministic_baseline": compact_baseline,
            "active_pair_performance": self._compact_pair_performance(active_name, performance_metrics)
            if active_name
            else {},
            "active_pair_recent_rank_context": self._recent_rank_for_pair(performance_metrics, active_name),
            "active_vs_rv22_naive_scaling": self._pair_vs_benchmark(active_name, performance_metrics)
            if active_name
            else {},
            "global_regime_monitor": {
                "current_global_regime": self._current_global_regime(market_features),
                "previous_global_regime": self._last_global_regime,
                "changed": global_change_reason is not None,
                "strong_switch_signal": global_change_reason is not None,
                "change_reason": global_change_reason or "",
            },
            "market_features": {
                key: market_features.get(key)
                for key in (
                    "rolling_vol",
                    "rule_vol_regime",
                    "vol_regime",
                    "intraday_realized_vol",
                )
                if key in market_features
            },
        }
        if self.sensitivity_enabled:
            payload["sensitiveness"] = self._sensitiveness_context()
            payload["sensitivity_switch_context"] = self._sensitivity_switch_context(performance_metrics)
        if not self.disable_regime_context:
            payload["predicted_regime"] = market_features.get("vol_regime")
            payload["ai_regime"] = market_features.get("ai_regime", {})
        else:
            payload["global_regime_monitor"] = {
                "changed": False,
                "change_reason": "regime_context_disabled",
            }
        if self.disable_deterministic_baseline_context:
            payload.pop("deterministic_baseline", None)
        if self.disable_benchmark_context:
            payload.pop("active_vs_rv22_naive_scaling", None)
            payload.pop("rv22_naive_scaling_benchmark", None)
        if self.disable_recent_rank_context:
            payload.pop("active_pair_recent_rank_context", None)
            payload.pop("recent_pair_rankings_100_60_20", None)
        return json.dumps(payload, default=str)

    def _current_global_regime(self, market_features: dict[str, Any]) -> str:
        rv = market_features.get("intraday_realized_vol", {})
        if isinstance(rv, dict):
            for key in ("regime", "vol_regime", "current_regime"):
                if key in rv:
                    regime = self._canonical_regime(rv.get(key))
                    return "middle" if regime == "mid" else regime
        regime = self._canonical_regime(market_features.get("rule_vol_regime", market_features.get("vol_regime")))
        return "middle" if regime == "mid" else regime

    def _global_regime_event_reason(self, market_features: dict[str, Any]) -> str | None:
        if self.disable_regime_context:
            return None
        rv = market_features.get("intraday_realized_vol", {})
        if isinstance(rv, dict) and bool(rv.get("regime_changed", False)):
            return "global_intraday_regime_changed"

        current = self._current_global_regime(market_features)
        if self._last_global_regime is not None and current != self._last_global_regime:
            return "global_regime_changed"
        return None

    def _should_call_selection(self, market_features: dict[str, Any]) -> tuple[bool, str]:
        if not self.ai_enabled:
            return False, "ai_disabled"
        if self.always_call_first and self._last_ai_selection_step is None:
            return True, "first_selection"
        if self._last_ai_selection_step is None:
            return True, "first_selection"
        gap = self._step - self._last_ai_selection_step
        if gap >= self.ai_selection_interval:
            return True, f"selection_interval_{self.ai_selection_interval}"
        return False, f"hold_selection_{gap}"

    def _fallback_candidate(self, candidates: list[dict[str, Any]]) -> str | None:
        if candidates:
            return str(candidates[0]["name"])
        return self._active_pair.name if self._active_pair is not None else None

    def _candidate_metric_score(self, candidate: dict[str, Any]) -> float:
        recent = candidate.get("recent_performance") or {}
        overall = candidate.get("overall_rank_context") or {}
        regime = candidate.get("regime_history") or {}
        active_regime = candidate.get("active_regime_ranks") or {}

        sharpe = self._safe_float(
            recent.get(
                "rolling_sharpe",
                active_regime.get("sharpe", regime.get("sharpe", overall.get("sharpe"))),
            ),
            0.0,
        )
        drawdown = abs(
            self._safe_float(
                recent.get(
                    "drawdown",
                    active_regime.get(
                        "max_drawdown",
                        regime.get("max_drawdown", overall.get("drawdown")),
                    ),
                ),
                0.0,
            )
        )
        cvar = abs(
            self._safe_float(
                recent.get("cvar_95", recent.get("tail_loss", overall.get("cvar_95"))),
                0.0,
            )
        )

        trailing_drawdowns = []
        trailing_sharpes = []
        trailing_cvars = []
        for key in ("trailing_10d", "trailing_63d", "trailing_126d"):
            block = recent.get(key)
            if not isinstance(block, dict):
                continue
            if "drawdown" in block:
                trailing_drawdowns.append(abs(self._safe_float(block.get("drawdown"), 0.0)))
            if "rolling_sharpe" in block:
                trailing_sharpes.append(self._safe_float(block.get("rolling_sharpe"), 0.0))
            tail_value = block.get("cvar_95", block.get("tail_loss"))
            if tail_value is not None:
                trailing_cvars.append(abs(self._safe_float(tail_value, 0.0)))

        if trailing_sharpes:
            sharpe = 0.55 * sharpe + 0.45 * (sum(trailing_sharpes) / len(trailing_sharpes))
        if trailing_cvars:
            cvar = max(cvar, sum(trailing_cvars) / len(trailing_cvars))

        if trailing_drawdowns:
            mean_dd = sum(trailing_drawdowns) / len(trailing_drawdowns)
            dd_std = (
                sum((value - mean_dd) ** 2 for value in trailing_drawdowns) / len(trailing_drawdowns)
            ) ** 0.5
            target_gap = abs(mean_dd - 0.06)
            max_excess = max(max(trailing_drawdowns) - 0.10, 0.0)
            consistency_penalty = dd_std + target_gap + max_excess
            drawdown = max(drawdown, mean_dd)
        else:
            consistency_penalty = abs(drawdown - 0.06) + max(drawdown - 0.10, 0.0)

        weights = {
            "very_low": (0.80, 4.75, 3.25, 2.25),
            "low": (0.90, 4.25, 2.75, 1.75),
            "normal": (1.00, 3.50, 2.00, 1.25),
            "high": (1.15, 2.75, 1.40, 0.80),
            "very_high": (1.25, 2.25, 1.00, 0.50),
        }[self.sensitiveness if self.sensitivity_enabled else "normal"]
        sharpe_weight, drawdown_weight, cvar_weight, consistency_weight = weights
        return float(
            sharpe_weight * sharpe
            - drawdown_weight * drawdown
            - cvar_weight * cvar
            - consistency_weight * consistency_penalty
        )

    def _deterministic_switch_response(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        active_name = self._active_pair.name if self._active_pair is not None else ""
        non_active = [row for row in candidates if str(row.get("name")) != active_name]
        if not active_name or not non_active:
            return {"action": "hold", "source": "deterministic_switch"}

        active_row = next((row for row in candidates if str(row.get("name")) == active_name), None)
        active_score = self._candidate_metric_score(active_row or {})
        sensitivity_context = (
            self._sensitivity_switch_context(performance_metrics, candidates)
            if self.sensitivity_enabled
            else {}
        )
        scored_non_active = sorted(
            (
                (self._candidate_metric_score(row), str(row.get("name")), row)
                for row in non_active
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best_score, best_name, _ = scored_non_active[0]
        regime_changed = self._global_regime_event_reason(market_features) is not None
        active_perf = performance_metrics.get(active_name, {})
        active_drawdown = abs(
            self._safe_float(active_perf.get("drawdown"), 0.0)
            if isinstance(active_perf, dict)
            else 0.0
        )
        switch_gap = {
            "very_low": 0.12,
            "low": 0.05,
            "normal": -0.02,
            "high": -0.10,
            "very_high": -0.18,
        }[self.sensitiveness if self.sensitivity_enabled else "normal"]
        drawdown_trigger = {
            "very_low": 0.16,
            "low": 0.12,
            "normal": 0.08,
            "high": 0.06,
            "very_high": 0.04,
        }[self.sensitiveness if self.sensitivity_enabled else "normal"]
        drawdown_gap = {
            "very_low": 0.08,
            "low": 0.00,
            "normal": -0.10,
            "high": -0.20,
            "very_high": -0.30,
        }[self.sensitiveness if self.sensitivity_enabled else "normal"]
        short_term_switch = bool(
            self.sensitivity_enabled
            and
            self.sensitiveness in {"high", "very_high"}
            and sensitivity_context.get("short_term_challenger_count", 0) >= 1
        )
        if self.sensitivity_enabled:
            hold_gate_open = bool(
                sensitivity_context.get("minimum_hold_satisfied", True)
                or sensitivity_context.get("group_exception", False)
            )
        else:
            hold_gate_open = int(self._active_since) >= self.switch_min_hold_days
        regime_switch = bool(regime_changed) and (
            (
                self.sensitivity_enabled
                and self.sensitiveness in {"high", "very_high"}
            )
            or best_score
            >= active_score
            + {
                "very_low": 0.10,
                "low": 0.03,
                "normal": -0.02,
            }.get(self.sensitiveness if self.sensitivity_enabled else "normal", -0.10)
        )
        should_switch = (
            short_term_switch
            or (
                hold_gate_open
                and (
                    regime_switch
                    or bool(best_name and best_score >= active_score + switch_gap)
                    or bool(
                        active_drawdown > drawdown_trigger
                        and best_name
                        and best_score >= active_score + drawdown_gap
                    )
                )
            )
        )
        out = {
            "action": "switch" if should_switch else "hold",
            "source": "deterministic_switch",
            "hold_days": int(self._active_since),
            "minimum_hold_days": self.switch_min_hold_days if not self.sensitivity_enabled else None,
            "minimum_hold_satisfied": hold_gate_open,
            "active_score": round(active_score, 6),
            "best_non_active_pair": best_name,
            "best_non_active_score": round(best_score, 6),
        }
        if self.sensitivity_enabled:
            out["sensitiveness"] = self.sensitiveness
            out["sensitivity_switch_context"] = sensitivity_context
        return out

    def _normalize_switch_decision_response(
        self,
        response: dict[str, Any],
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        action = self._find_action_value(response)
        if action:
            out = dict(response)
            out["action"] = "hold" if action == "stay" else action
            if out["action"] == "review":
                out["action"] = "switch"
            if out["action"] == "change":
                out["action"] = "switch"
            return out

        fallback = self._deterministic_switch_response(market_features, performance_metrics, candidates)
        if self._global_regime_event_reason(market_features) is not None:
            active_name = self._active_pair.name if self._active_pair is not None else ""
            if any(str(row.get("name")) != active_name for row in candidates):
                fallback = {"action": "switch", "source": "missing_ai_action_regime_event_fallback"}
        fallback["raw_missing_action_response"] = response
        return fallback

    def _deterministic_selection_response(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not candidates:
            return {
                "action": "hold",
                "pair": self._active_pair.name if self._active_pair is not None else self.default_pair.name,
                "source": "deterministic_selection",
            }
        scored = sorted(
            (
                (self._candidate_metric_score(row), str(row.get("name")), row)
                for row in candidates
            ),
            key=lambda item: (-item[0], item[1]),
        )
        chosen_score, _, chosen = scored[0]
        out = {
            "action": "switch",
            "pair": str(chosen["name"]),
            "source": "deterministic_selection",
            "score": round(chosen_score, 6),
            "score_rule": "Sharpe/drawdown/CVaR/drawdown-consistency score",
        }
        if self.sensitivity_enabled:
            out["sensitiveness"] = self.sensitiveness
            out["score_rule"] = "sensitiveness-adjusted Sharpe/drawdown/CVaR/drawdown-consistency score"
        return out

    def _choose_with_ai(
        self,
        market_features: dict[str, Any],
        performance_metrics: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[StrategyPair, bool, dict[str, Any], str, str]:
        active_name = self._active_pair.name if self._active_pair is not None else None
        fallback_name = active_name or self._fallback_candidate(candidates)
        if fallback_name is None:
            fallback_name = self.default_pair.name

        should_call, call_reason = self._should_call_selection(market_features)
        response: dict[str, Any] = {}
        error = ""
        chosen_name = fallback_name
        used_ai = False
        valid_names = {str(row["name"]) for row in candidates}
        if active_name:
            valid_names.add(active_name)

        if should_call:
            phase = "switch"
            try:
                if self.deterministic_switch_decision:
                    decision_response = self._deterministic_switch_response(
                        market_features,
                        performance_metrics,
                        candidates,
                    )
                else:
                    decision_response = self._call_ai_switch_decision(
                        self._switch_decision_payload(market_features, performance_metrics)
                    )
                decision_response = self._normalize_switch_decision_response(
                    decision_response,
                    market_features,
                    performance_metrics,
                    candidates,
                )
                sensitivity_context = (
                    self._sensitivity_switch_context(performance_metrics, candidates)
                    if self.sensitivity_enabled
                    else {}
                )
                decision_action = str(decision_response.get("action", "hold")).lower()
                if (
                    self.sensitivity_enabled
                    and
                    decision_action in ("hold", "stay")
                    and self.sensitiveness in {"high", "very_high"}
                    and int(sensitivity_context.get("short_term_challenger_count", 0)) >= 1
                ):
                    decision_response = {
                        **decision_response,
                        "action": "switch",
                        "source": str(decision_response.get("source", "ai_switch_decision"))
                        + "_short_term_challenger_gate",
                        "sensitivity_switch_context": sensitivity_context,
                    }
                    decision_action = "switch"
                if (
                    self.sensitivity_enabled
                    and
                    decision_action in ("switch", "review", "change")
                    and self.sensitiveness in {"normal", "low", "very_low"}
                    and not bool(sensitivity_context.get("minimum_hold_satisfied", True))
                    and not bool(sensitivity_context.get("group_exception", False))
                ):
                    decision_response = {
                        **decision_response,
                        "action": "hold",
                        "source": str(decision_response.get("source", "ai_switch_decision"))
                        + "_minimum_hold_gate",
                        "sensitivity_switch_context": sensitivity_context,
                    }
                    decision_action = "hold"
                elif self.sensitivity_enabled and decision_action in ("switch", "review", "change"):
                    decision_response = {
                        **decision_response,
                        "sensitivity_switch_context": sensitivity_context,
                    }
                elif (
                    not self.sensitivity_enabled
                    and decision_action in ("switch", "review", "change")
                    and int(self._active_since) < self.switch_min_hold_days
                ):
                    decision_response = {
                        **decision_response,
                        "action": "hold",
                        "source": str(decision_response.get("source", "ai_switch_decision"))
                        + "_minimum_hold_gate",
                        "hold_days": int(self._active_since),
                        "minimum_hold_days": self.switch_min_hold_days,
                    }
                    decision_action = "hold"
                response = {"decision": decision_response}
                if decision_action in ("switch", "review", "change"):
                    switch_candidates = [
                        row for row in candidates if str(row.get("name")) != str(active_name)
                    ]
                    if not switch_candidates:
                        action = "hold"
                        chosen_name = active_name or self._fallback_candidate(candidates) or self.default_pair.name
                        response["selection"] = {
                            "action": action,
                            "pair": chosen_name,
                            "reason": "No non-active switch candidates were available",
                            "confidence": 0.0,
                        }
                    else:
                        preferred_names: list[str] = []
                        if self.sensitivity_enabled:
                            if self.sensitiveness in {"high", "very_high"}:
                                preferred_names = [
                                    str(row.get("pair"))
                                    for row in sensitivity_context.get("top_short_term_challengers", [])
                                    if isinstance(row, dict)
                                ]
                            else:
                                preferred_names = [
                                    str(row.get("pair"))
                                    for row in sensitivity_context.get("top_multi_scale_challengers", [])
                                    if isinstance(row, dict)
                                ]
                        if preferred_names:
                            preferred = [
                                row for row in switch_candidates if str(row.get("name")) in set(preferred_names)
                            ]
                            preferred.sort(
                                key=lambda row: preferred_names.index(str(row.get("name")))
                                if str(row.get("name")) in preferred_names
                                else 10**9
                            )
                            rest = [
                                row for row in switch_candidates if str(row.get("name")) not in set(preferred_names)
                            ]
                            switch_candidates = (
                                preferred
                                + self._limit_selection_candidates(rest)[
                                    : max(self.candidate_top_n - len(preferred), 0)
                                ]
                            )[: self.candidate_top_n]
                        else:
                            switch_candidates = self._limit_selection_candidates(switch_candidates)
                        switch_valid_names = {str(row["name"]) for row in switch_candidates}
                        phase = "selection"
                        if self.deterministic_pair_selection:
                            selection_response = self._deterministic_selection_response(switch_candidates)
                        else:
                            selection_response = self._call_ai_selection(
                                self._selection_payload(
                                    market_features,
                                    performance_metrics,
                                    switch_candidates,
                                )
                            )
                        try:
                            action, chosen_name = self._selection_choice_from_response(
                                selection_response,
                                switch_valid_names,
                                active_name,
                                switch_candidates,
                            )
                        except Exception:
                            selection_response = {
                                **self._deterministic_selection_response(switch_candidates),
                                "source": "invalid_ai_selection_non_active_fallback",
                                "raw_ai_selection_response": selection_response,
                            }
                            action, chosen_name = self._selection_choice_from_response(
                                selection_response,
                                switch_valid_names,
                                active_name,
                                switch_candidates,
                            )
                        if action in ("hold", "stay") or chosen_name == active_name:
                            selection_response = {
                                **self._deterministic_selection_response(switch_candidates),
                                "source": "missing_ai_selection_non_active_fallback",
                                "raw_ai_selection_response": selection_response,
                            }
                            action, chosen_name = self._selection_choice_from_response(
                                selection_response,
                                switch_valid_names,
                                active_name,
                                switch_candidates,
                            )
                        response["selection"] = selection_response
                else:
                    action = "hold"
                    chosen_name = active_name or self._fallback_candidate(candidates) or self.default_pair.name
                used_ai = True
                self._last_ai_selection = {
                    "action": action,
                    "pair": chosen_name,
                    "reason": str(
                        response.get("selection", response.get("decision", {})).get("reason", "")
                    )[:240],
                    "confidence": self._safe_float(
                        response.get("selection", response.get("decision", {})).get("confidence"),
                        0.0,
                    ),
                    "call_reason": call_reason,
                    "error": "",
                }
                self._last_ai_selection_step = self._step
            except Exception as exc:
                if phase == "switch":
                    decision_response = self._normalize_switch_decision_response(
                        {},
                        market_features,
                        performance_metrics,
                        candidates,
                    )
                    response = {"decision": decision_response}
                    action = str(decision_response.get("action", "hold")).lower()
                    if action in ("switch", "review", "change"):
                        switch_candidates = [
                            row for row in candidates if str(row.get("name")) != str(active_name)
                        ]
                        switch_candidates = self._limit_selection_candidates(switch_candidates)
                        if switch_candidates:
                            selection_response = self._deterministic_selection_response(switch_candidates)
                            _, chosen_name = self._selection_choice_from_response(
                                selection_response,
                                {str(row["name"]) for row in switch_candidates},
                                active_name,
                                switch_candidates,
                            )
                            response["selection"] = selection_response
                        else:
                            chosen_name = (
                                active_name
                                or self._fallback_candidate(candidates)
                                or self.default_pair.name
                            )
                            action = "hold"
                    else:
                        chosen_name = (
                            active_name
                            or self._fallback_candidate(candidates)
                            or self.default_pair.name
                        )
                    error = str(exc)
                    used_ai = True
                    self._last_ai_selection = {
                        "action": action,
                        "pair": chosen_name,
                        "reason": "Switch decision failed; used deterministic fallback for this checkpoint",
                        "confidence": 0.0,
                        "call_reason": call_reason,
                        "error": error,
                    }
                    if not self.retry_ai_after_failure:
                        self._last_ai_selection_step = self._step
                    if not self.fail_open:
                        raise
                    pair = next(
                        (candidate for candidate in self.pairs if candidate.name == chosen_name),
                        self.default_pair,
                    )
                    return pair, used_ai, response, error, call_reason

                recovered = self._fallback_selection_from_raw_response(valid_names, active_name, candidates)
                if recovered is not None:
                    response = recovered
                    action, chosen_name = self._selection_choice_from_response(
                        response,
                        valid_names,
                        active_name,
                        candidates,
                    )
                    used_ai = True
                    self._last_ai_selection = {
                        "action": action,
                        "pair": chosen_name,
                        "reason": str(response.get("reason", ""))[:240],
                        "confidence": self._safe_float(response.get("confidence"), 0.0),
                        "call_reason": call_reason,
                        "error": "",
                    }
                    self._last_ai_selection_step = self._step
                else:
                    error = str(exc)
                    self._last_ai_selection = {
                        "action": "hold",
                        "pair": chosen_name,
                        "reason": "AI selection failed; held fallback until next scheduled review",
                        "confidence": 0.0,
                        "call_reason": call_reason,
                        "error": error,
                    }
                    if not self.retry_ai_after_failure:
                        self._last_ai_selection_step = self._step
                    if not self.fail_open:
                        raise

        pair = next((candidate for candidate in self.pairs if candidate.name == chosen_name), self.default_pair)
        return pair, used_ai, response, error, call_reason

    def _selection_choice_from_response(
        self,
        response: dict[str, Any],
        valid_names: set[str],
        active_name: str | None,
        candidates: list[dict[str, Any]],
    ) -> tuple[str, str]:
        action = self._find_action_value(response) or str(response.get("action", "")).lower()
        requested = self._find_pair_value(response, valid_names) or str(response.get("pair", active_name or ""))
        if not action and requested in valid_names:
            action = "switch"
        if not action:
            action = "hold"
        if action in ("hold", "stay"):
            return action, active_name or self._fallback_candidate(candidates) or self.default_pair.name
        if active_name and requested == active_name:
            return "hold", active_name
        if requested not in valid_names:
            raise ValueError(f"AI selected unknown or unshown pair: {requested!r}")
        return action, requested

    def _fallback_selection_from_raw_response(
        self,
        valid_names: set[str],
        active_name: str | None,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not isinstance(self._last_raw_response, dict):
            return None
        choices = self._last_raw_response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            return None
        text = " ".join(
            str(message.get(key, ""))
            for key in ("reasoning_content", "reasoning")
            if isinstance(message.get(key), str)
        ).lower()
        if not text:
            return None
        for name in sorted(valid_names, key=len, reverse=True):
            lowered = name.lower()
            switch_patterns = (
                f"switch to {lowered}",
                f"switch into {lowered}",
                f"switch pair to {lowered}",
                f"select {lowered}",
                f"choose {lowered}",
            )
            if any(pattern in text for pattern in switch_patterns):
                return {
                    "action": "switch",
                    "pair": name,
                    "reason": "Recovered clear switch target from provider reasoning-only response",
                    "confidence": 0.0,
                    "recovered_from_raw_response": True,
                }
        return {
            "action": "hold",
            "pair": active_name or self._fallback_candidate(candidates) or self.default_pair.name,
            "reason": "Held after provider returned reasoning-only response without final JSON",
            "confidence": 0.0,
            "recovered_from_raw_response": True,
        }

    def select(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        timestamp: Any = None,
    ) -> StrategyPair:
        enriched_features, ai_regime, call_reason = self._with_ai_regime(
            market_features,
            performance_metrics,
            timestamp=timestamp,
        )
        if self._active_pair is None:
            self._active_pair = self.default_pair

        should_call_selection, _ = self._should_call_selection(enriched_features)
        candidates = (
            self._regime_pair_pool(
                str(enriched_features.get("vol_regime", "middle")),
                performance_metrics,
            )
            if should_call_selection or self._active_pair is None
            else []
        )
        chosen_pair, used_ai, response, error, selection_reason = self._choose_with_ai(
            enriched_features,
            performance_metrics,
            candidates,
        )

        can_switch = (self._active_since + 1) >= self.sticky_period
        switched = False
        if can_switch and chosen_pair.name != self._active_pair.name:
            self._active_pair = chosen_pair
            self._active_since = 0
            switched = True
        else:
            self._active_since += 1
        self._switch_history.append(1 if switched else 0)
        self._last_global_regime = self._current_global_regime(enriched_features)

        scores: dict[str, float] = {}
        score_components: dict[str, dict[str, float]] = {}
        for pair in self.pairs:
            components = self._score_components(pair, enriched_features, diagnostics, performance_metrics)
            scores[pair.name] = components["total"]
            score_components[pair.name] = components

        decision = {
            "timestamp": timestamp,
            "selected_pair": self._active_pair.name,
            "selected_estimator": self._active_pair.estimator_name,
            "selected_controller": self._active_pair.controller_name,
            "switched": switched,
            "scores": scores,
            "score_components": score_components,
            "market_features": dict(enriched_features),
            "performance_metrics": dict(performance_metrics),
            "ai_regime": ai_regime,
            "ai_regime_call_reason": call_reason,
            "ai_regime_call_count": self._ai_regime_call_count,
            "ai_selection_used": used_ai,
            "ai_selection_response": response,
            "ai_selection_error": error,
            "ai_selection_call_reason": selection_reason,
            "ai_selection_call_count": self._ai_selection_call_count,
            "regime_pair_pool": candidates,
        }
        self.decisions.append(decision)
        self._step += 1
        return self._active_pair


__all__ = ["AIRegimeRouter"]
