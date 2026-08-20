"""Generate and persist an AI-predicted volatility-regime time series.

The output is a daily CSV that can be reused by AIRegimeRouter through
``precomputed_ai_regime_path`` so future router runs do not pay for regime
classification calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.evaluate_router_protocol import _build_market_features, _load_data  # noqa: E402
from src.router.ai_regime_router import AIRegimeRouter  # noqa: E402
from src.router.strategy_pair import StrategyPair  # noqa: E402


class _DummyEstimator:
    pass


class _DummyController:
    pass


def load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _canonical_regime(value: Any) -> str:
    text = str(value).lower()
    if text in ("mid", "middle", "normal"):
        return "middle"
    if text in ("low", "high"):
        return text
    return "middle"


def _load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing = pd.read_csv(path)
    if "prediction_date" not in existing.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in existing.drop_duplicates("prediction_date").iterrows():
        prediction_date = str(row["prediction_date"])[:10]
        regime = _canonical_regime(row.get("ai_vol_regime", row.get("vol_regime", "middle")))
        out[prediction_date] = {
            "prediction_date": prediction_date,
            "feature_end_date": str(row.get("feature_end_date", ""))[:10]
            if "feature_end_date" in existing.columns
            else "",
            "ai_vol_regime": regime,
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "reason": str(row.get("reason", "")),
            "error": str(row.get("error", "")) if pd.notna(row.get("error", "")) else "",
            "raw_response": str(row.get("raw_response", "")) if pd.notna(row.get("raw_response", "")) else "",
        }
    return out


def _prediction_dates(index: pd.DatetimeIndex, roll_window: int, interval: int) -> list[pd.Timestamp]:
    if len(index) <= roll_window + 1:
        return []
    return [index[i] for i in range(roll_window + 1, len(index), interval)]


def _materialize_daily_series(market_index: pd.DatetimeIndex, predictions: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sorted_anchors = sorted(pd.Timestamp(key) for key in predictions)
    active: dict[str, Any] | None = None
    anchor_pos = 0
    for date in market_index:
        while anchor_pos < len(sorted_anchors) and sorted_anchors[anchor_pos] <= date:
            active = predictions[str(sorted_anchors[anchor_pos].date())]
            anchor_pos += 1
        if active is None:
            continue
        rows.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "ai_vol_regime": active["ai_vol_regime"],
                "confidence": active["confidence"],
                "reason": active["reason"],
                "prediction_date": active["prediction_date"],
                "feature_end_date": active.get("feature_end_date", ""),
                "error": active["error"],
                "raw_response": active["raw_response"],
            }
        )
    return pd.DataFrame(rows)


def _write_daily_series(output_path: Path, market_index: pd.DatetimeIndex, predictions: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = _materialize_daily_series(market_index, predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def _strictly_prior_feature_window(
    market_df: pd.DataFrame,
    anchor: pd.Timestamp,
    roll_window: int,
) -> tuple[pd.Series, pd.Timestamp]:
    anchor_pos = int(market_df.index.searchsorted(anchor))
    if anchor_pos < roll_window:
        raise ValueError(f"Not enough prior observations before {anchor.date()} for roll_window={roll_window}")
    feature_end = market_df.index[anchor_pos - 1]
    window = market_df["returns_clean"].iloc[anchor_pos - roll_window : anchor_pos]
    return window, pd.Timestamp(feature_end)


def _attach_intraday_features(features: dict[str, Any], intraday: pd.DataFrame | None, date: pd.Timestamp) -> None:
    if intraday is None or intraday.empty:
        return
    key = pd.Timestamp(date).tz_localize(None).normalize()
    if key not in intraday.index:
        return
    row = intraday.loc[key]
    features["intraday_realized_vol"] = {
        "realized_vol": float(row.get("realized_vol", 0.0) or 0.0),
        "n_obs": int(float(row.get("n_obs", 0.0) or 0.0)),
        "coverage": float(row.get("coverage", 0.0) or 0.0),
    }


def _build_router(args: argparse.Namespace) -> AIRegimeRouter:
    pair = StrategyPair(
        name="regime_generation_dummy",
        estimator=_DummyEstimator(),
        controller=_DummyController(),
    )
    return AIRegimeRouter(
        pairs=[pair],
        params={
            "provider": args.provider,
            "api_format": "chat_completions",
            "model": args.model,
            "api_key": args.api_key or os.getenv("NVAPI_KEY") or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
            "max_total_ai_calls": args.max_calls,
            "max_ai_regime_calls": args.max_calls,
            "max_ai_selection_calls": 0,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "timeout": args.timeout,
            "fail_open": False,
        },
    )


def generate_series(args: argparse.Namespace) -> pd.DataFrame:
    load_env_file(args.env_file)
    load_env_file(args.extra_env_file)

    with open(args.strategy, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    market_df = _load_data(cfg)
    roll_window = int(args.roll_window or cfg.get("roll_window", 252))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intraday = None
    if args.intraday_path:
        intraday_path = Path(args.intraday_path)
        if intraday_path.exists():
            intraday = pd.read_parquet(intraday_path)
            intraday.index = pd.to_datetime(intraday.index).tz_localize(None).normalize()
            intraday = intraday.sort_index()

    router = _build_router(args)
    existing = _load_existing_predictions(output_path) if args.resume else {}
    predictions: dict[str, dict[str, Any]] = dict(existing)
    if getattr(args, "start_date", None):
        start_date = pd.Timestamp(args.start_date)
        start_pos = int(market_df.index.searchsorted(start_date, side="left"))
        start_pos = max(start_pos, roll_window)
        anchors = [market_df.index[i] for i in range(start_pos, len(market_df.index), args.interval)]
    else:
        anchors = _prediction_dates(market_df.index, roll_window=roll_window, interval=args.interval)
    if getattr(args, "end_date", None):
        end_date = pd.Timestamp(args.end_date)
        anchors = [anchor for anchor in anchors if anchor <= end_date]

    for anchor in anchors:
        prediction_date = str(anchor.date())
        if prediction_date in predictions:
            continue
        if args.max_calls > 0 and router._ai_regime_call_count >= args.max_calls:
            break

        feature_window, feature_end = _strictly_prior_feature_window(
            market_df,
            anchor,
            roll_window=roll_window,
        )
        features = _build_market_features(feature_window)
        features["feature_end_date"] = str(feature_end.date())
        features["prediction_effective_date"] = prediction_date
        _attach_intraday_features(features, intraday, feature_end)
        prompt = router._ai_regime_payload(features, {})
        try:
            response = None
            last_error: Exception | None = None
            for attempt in range(args.retries + 1):
                try:
                    response = router._call_ai_regime(prompt)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= args.retries:
                        raise
                    time.sleep(args.retry_sleep_seconds)
            if response is None:
                raise RuntimeError(str(last_error or "AI regime call failed"))
            regime = _canonical_regime(response.get("vol_regime", "middle"))
            predictions[prediction_date] = {
                "prediction_date": prediction_date,
                "feature_end_date": str(feature_end.date()),
                "ai_vol_regime": regime,
                "confidence": float(response.get("confidence", 0.0) or 0.0),
                "reason": str(response.get("reason", ""))[:240],
                "error": "",
                "raw_response": json.dumps(response, default=str),
            }
            _write_daily_series(output_path, market_df.index, predictions)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        except Exception as exc:
            predictions[prediction_date] = {
                "prediction_date": prediction_date,
                "feature_end_date": str(feature_end.date()) if "feature_end" in locals() else "",
                "ai_vol_regime": args.failure_regime,
                "confidence": 0.0,
                "reason": "AI regime generation failed",
                "error": str(exc),
                "raw_response": "",
            }
            _write_daily_series(output_path, market_df.index, predictions)
            if args.fail_closed:
                raise

    return _write_daily_series(output_path, market_df.index, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reusable AI volatility-regime series.")
    parser.add_argument("--strategy", default="configs/strategies/router_master.yaml")
    parser.add_argument("--output", default="results/evaluation/ai_regime_series/ai_regime_10d.csv")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--roll-window", type=int, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-calls", type=int, default=365)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--retry-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--intraday-path", default="data/processed/SP500_Intraday_RealizedVol.parquet")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--extra-env-file", default="tests/.env")
    parser.add_argument("--failure-regime", default="middle", choices=["low", "middle", "high"])
    parser.add_argument("--fail-closed", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    out = generate_series(args)
    print(
        json.dumps(
            {
                "output": args.output,
                "rows": int(len(out)),
                "prediction_count": int(out["prediction_date"].nunique()) if not out.empty else 0,
                "error_rows": int(out["error"].astype(str).ne("").sum()) if "error" in out else 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
