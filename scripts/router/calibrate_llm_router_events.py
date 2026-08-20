"""Ask an AI to calibrate LLM-router intraday-RV event thresholds.

This is intentionally not a grid search. The script summarizes historical
intraday-derived realized volatility behavior, then asks a model for one
recommended event policy. Stage 1 is AI-tuned; the code does not search
threshold combinations or score candidate policies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.env import Env


SYSTEM_PROMPT = """You calibrate event triggers for a volatility-targeting LLM router.
Recommend when the router should ask an AI to reconsider the estimator-controller pair.
Do not overfit. Prefer stable, interpretable thresholds that avoid excessive switching.
Return strict JSON only with keys:
{
  "decision_mode": "intraday_rv_event",
  "min_decision_gap": 21,
  "rv_zscore_trigger": 1.5,
  "rv_change_trigger": 0.25,
  "rv_percentile_trigger": 0.9,
  "rationale": "...",
  "expected_behavior": "...",
  "risks": ["..."]
}
"""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = Env.path("root") / path
    return path


def _load_intraday_rv(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    df.index = df.index.normalize()
    df = df.sort_index()
    if "realized_vol" not in df.columns:
        raise ValueError(f"{path} must contain a realized_vol column.")
    return df


def _event_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    rv = pd.to_numeric(df["realized_vol"], errors="coerce").astype(float)
    out = pd.DataFrame(index=df.index)
    out["realized_vol"] = rv
    out["pct_change_1d"] = rv.pct_change().replace([np.inf, -np.inf], np.nan)
    out["rolling_mean"] = rv.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    out["rolling_std"] = rv.rolling(lookback, min_periods=max(5, lookback // 3)).std(ddof=1)
    out["zscore"] = (rv - out["rolling_mean"]) / out["rolling_std"]
    out["slope_5d"] = rv - rv.shift(5)
    out["percentile"] = rv.rolling(lookback, min_periods=max(5, lookback // 3)).rank(pct=True)
    if "coverage" in df.columns:
        out["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["realized_vol"])


def _quantiles(series: pd.Series, qs: list[float]) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {f"q{int(q * 100):02d}": float(s.quantile(q)) for q in qs}


def build_summary(df: pd.DataFrame, lookback: int) -> dict[str, Any]:
    features = _event_features(df, lookback)
    clean = features.dropna(subset=["zscore", "pct_change_1d", "percentile"])
    coverage = features["coverage"].dropna() if "coverage" in features.columns else pd.Series(dtype=float)

    return {
        "date_start": str(features.index.min().date()),
        "date_end": str(features.index.max().date()),
        "n_days": int(len(features)),
        "lookback": int(lookback),
        "realized_vol_quantiles": _quantiles(features["realized_vol"], [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
        "abs_pct_change_quantiles": _quantiles(features["pct_change_1d"].abs(), [0.5, 0.75, 0.9, 0.95, 0.99]),
        "abs_zscore_quantiles": _quantiles(features["zscore"].abs(), [0.5, 0.75, 0.9, 0.95, 0.99]),
        "slope_5d_quantiles": _quantiles(features["slope_5d"], [0.05, 0.25, 0.5, 0.75, 0.95]),
        "coverage_quantiles": _quantiles(coverage, [0.05, 0.25, 0.5, 0.75, 0.95]) if not coverage.empty else {},
        "valid_feature_days": int(len(clean)),
        "recent_curve_tail": [float(x) for x in features["realized_vol"].tail(21).round(6).tolist()],
    }


def build_prompt(summary: dict[str, Any]) -> str:
    return (
        "Historical intraday-derived realized volatility summary:\n"
        f"{json.dumps(summary, indent=2)}\n\n"
        "Choose event thresholds for when an LLM router should reconsider the current pair. "
        "Target roughly monthly to quarterly AI reviews in calm markets, but allow faster review during true volatility shocks. "
        "Avoid daily calls and avoid one-pair lock-in caused by stale regimes. "
        "This is not a grid search: infer one stable policy directly from the distribution, z-score behavior, "
        "one-day RV changes, slopes, coverage, and recent RV curve."
    )


def _extract_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model response was not JSON: {text!r}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


def call_model(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    provider = str(args.provider).lower()
    api_key = args.api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NVAPI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key provided. Set NVIDIA_API_KEY/NVAPI_KEY/OPENAI_API_KEY or use --api-key.")

    if provider == "nvidia":
        endpoint = args.endpoint or "https://integrate.api.nvidia.com/v1/chat/completions"
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": args.max_output_tokens,
            "temperature": args.temperature,
        }
    else:
        endpoint = args.endpoint or "https://api.openai.com/v1/responses"
        payload = {
            "model": args.model,
            "instructions": SYSTEM_PROMPT,
            "input": prompt,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
        }

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AI calibration request failed: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"AI calibration request failed: {exc.reason}") from exc

    return _parse_json(_extract_text(data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate LLM-router event thresholds with AI.")
    parser.add_argument("--rv-path", default="data/processed/SP500_Intraday_RealizedVol.parquet")
    parser.add_argument("--output-dir", default="results/evaluation/llm_router_event_calibration")
    parser.add_argument("--lookback", type=int, default=21)
    parser.add_argument("--provider", default="nvidia", choices=["nvidia", "openai"])
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no-api", action="store_true", help="Only write the AI prompt and summary.")
    args = parser.parse_args()

    out_dir = _resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_intraday_rv(_resolve_path(args.rv_path))
    summary = build_summary(df, args.lookback)
    prompt = build_prompt(summary)

    with open(out_dir / "intraday_rv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "ai_calibration_prompt.md", "w", encoding="utf-8") as f:
        f.write(prompt)

    if args.no_api:
        print(f"Wrote summary and prompt to {out_dir}")
        return

    recommendation = call_model(prompt, args)
    with open(out_dir / "ai_event_policy_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)

    snippet = {
        "router": {
            "params": {
                "decision_mode": recommendation.get("decision_mode", "intraday_rv_event"),
                "min_decision_gap": int(_safe_float(recommendation.get("min_decision_gap"), 21)),
                "rv_zscore_trigger": _safe_float(recommendation.get("rv_zscore_trigger"), 1.5),
                "rv_change_trigger": _safe_float(recommendation.get("rv_change_trigger"), 0.25),
                "rv_percentile_trigger": _safe_float(recommendation.get("rv_percentile_trigger"), 0.9),
            }
        }
    }
    with open(out_dir / "router_event_policy.yaml", "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(snippet, f, sort_keys=False)

    print(json.dumps(recommendation, indent=2))
    print(f"Wrote calibration outputs to {out_dir}")


if __name__ == "__main__":
    main()
