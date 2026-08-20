"""Fast walk-forward router protocol using precomputed pair backtests.

This keeps the same nested train/test structure as evaluate_router_protocol.py,
but reads the already-generated estimator/controller pair parquet files instead
of re-running each estimator and controller inside every split.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.evaluate_router_protocol import (  # noqa: E402
    _build_market_features,
    _build_pairs,
    _build_router,
    _calmar,
    _cvar_95,
    _load_data,
    _max_drawdown,
    _qlike_var,
    _rolling_sharpe,
    _safe_float,
)
from src.env import Env  # noqa: E402


def _load_pair_results(cfg: dict, pair_results_dir: Path, max_pairs: int | None = None) -> dict[str, pd.DataFrame]:
    pairs_cfg = cfg.get("router", {}).get("pairs", [])
    out: dict[str, pd.DataFrame] = {}
    for i, pair_cfg in enumerate(pairs_cfg):
        if max_pairs is not None and i >= max_pairs:
            break
        name = pair_cfg.get("name", f"pair_{i}")
        path = pair_results_dir / f"{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = _normalize_pair_result(df)
        out[name] = df
    return out


def _normalize_pair_result(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "returns" not in out.columns:
        if "returns_with_rf" in out.columns:
            out["returns"] = out["returns_with_rf"]
        elif "returns_no_rf" in out.columns:
            out["returns"] = out["returns_no_rf"]
    if "equity_curve" not in out.columns:
        if "equity_curve_with_rf" in out.columns:
            out["equity_curve"] = out["equity_curve_with_rf"]
        elif "equity_curve_no_rf" in out.columns:
            out["equity_curve"] = out["equity_curve_no_rf"]
    if "asset_returns" not in out.columns and "asset_returns_clean" in out.columns:
        out["asset_returns"] = out["asset_returns_clean"]
    return out


def _ann_vol(returns: pd.Series | np.ndarray) -> float:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * math.sqrt(252.0))


def _pair_train_metrics(pair_df: pd.DataFrame, target_vol: float) -> dict[str, float]:
    r = pair_df["returns"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    asset = pair_df.get("asset_returns", pd.Series(index=pair_df.index, dtype=float)).astype(float)
    vol = pair_df.get("vol_estimate", pd.Series(index=pair_df.index, dtype=float)).astype(float)
    weight = pair_df.get("weight", pd.Series(index=pair_df.index, dtype=float)).astype(float)

    valid = vol.replace([np.inf, -np.inf], np.nan).notna() & asset.replace([np.inf, -np.inf], np.nan).notna()
    valid = valid & (vol > 0)
    if int(valid.sum()) >= 5:
        rv = np.sqrt(np.maximum(252.0 * asset[valid].to_numpy() ** 2, 1e-12))
        qlike = _qlike_var(rv**2, vol[valid].to_numpy() ** 2)
    else:
        qlike = np.nan

    equity = np.exp(r.cumsum()).to_numpy()
    turnover = weight.diff().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    realized_vol = _ann_vol(r)
    return {
        "qlike": float(qlike) if np.isfinite(qlike) else np.nan,
        "rmse_vol": float(np.sqrt(np.nanmean((vol[valid].to_numpy() - rv) ** 2))) if int(valid.sum()) >= 5 else np.nan,
        "turning_point_delay": np.nan,
        "kupiec_lr": np.nan,
        "christoffersen_lr": np.nan,
        "violation_rate": np.nan,
        "qlike_high": np.nan,
        "qlike_low": np.nan,
        "vol_tracking_error": abs(realized_vol - target_vol),
        "drawdown": _max_drawdown(equity),
        "tail_loss": _cvar_95(r.to_numpy()),
        "turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "cost_adjusted_sharpe": _rolling_sharpe(r.to_numpy()),
        "exposure_convexity": np.nan,
        "realized_vol": realized_vol,
    }


def _benchmark_metrics(benchmark_df: pd.DataFrame | None, end_pos: int, target_vol: float, lookback: int = 126) -> dict[str, float]:
    if benchmark_df is None or benchmark_df.empty:
        return {}
    hist = benchmark_df.iloc[max(0, end_pos - lookback) : end_pos].copy()
    if hist.empty:
        return {}
    returns_col = "returns"
    if returns_col not in hist.columns:
        if "returns_with_rf" in hist.columns:
            returns_col = "returns_with_rf"
        elif "returns_no_rf" in hist.columns:
            returns_col = "returns_no_rf"
    returns = pd.to_numeric(hist.get(returns_col, pd.Series(0.0, index=hist.index)), errors="coerce").fillna(0.0)
    weight = pd.to_numeric(hist.get("weight", pd.Series(0.0, index=hist.index)), errors="coerce").fillna(0.0)
    turnover = weight.diff().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    realized_vol = _ann_vol(returns)
    return {
        "name": "rv22_naive_scaling",
        "obs": int(len(returns)),
        "rolling_sharpe": _rolling_sharpe(returns.to_numpy()),
        "drawdown": _max_drawdown(np.exp(returns.cumsum()).to_numpy()),
        "realized_vol": realized_vol,
        "vol_tracking_error": abs(realized_vol - target_vol),
        "turnover": float(turnover.mean()) if len(turnover) else 0.0,
    }


def _pair_regime_history_metrics(
    pair_results: dict[str, pd.DataFrame],
    market_regimes: pd.Series,
    end_pos: int,
    target_vol: float,
    min_obs: int = 20,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {"low": {}, "middle": {}, "high": {}}
    for pair_name, pair_df in pair_results.items():
        hist = pair_df.iloc[:end_pos].copy()
        if hist.empty:
            continue
        if "vol_regime" in hist.columns:
            regimes = hist["vol_regime"].astype(str).str.lower().replace({"mid": "middle", "normal": "middle"})
        else:
            regimes = market_regimes.iloc[:end_pos].reindex(hist.index).astype(str)
        weight = hist.get("weight", pd.Series(0.0, index=hist.index)).astype(float)
        turnover = weight.diff().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for regime in ("low", "middle", "high"):
            mask = regimes.eq(regime)
            if int(mask.sum()) < min_obs:
                continue
            sub = hist.loc[mask]
            returns = pd.to_numeric(sub["returns"], errors="coerce").fillna(0.0)
            realized_vol = _ann_vol(returns)
            out[regime][pair_name] = {
                "n_days": int(len(returns)),
                "sharpe": _rolling_sharpe(returns.to_numpy()),
                "max_drawdown": _max_drawdown(np.exp(returns.cumsum()).to_numpy()),
                "avg_turnover": float(turnover.loc[sub.index].mean()),
                "realized_vol": realized_vol,
                "vol_tracking_error": abs(realized_vol - target_vol),
            }
    return out


def _historical_market_regimes(market_df: pd.DataFrame, roll_window: int) -> pd.Series:
    regimes = pd.Series("middle", index=market_df.index, dtype="object")
    returns = market_df["returns_clean"].astype(float)
    for i in range(roll_window, len(market_df)):
        features = _build_market_features(returns.iloc[i - roll_window : i])
        regime = str(features.get("vol_regime", "middle")).lower()
        regimes.iloc[i] = "middle" if regime in ("mid", "normal") else regime
    return regimes


def _metrics_to_score(metrics: dict[str, float]) -> float:
    return (
        _safe_float(metrics.get("cost_adjusted_sharpe"))
        - 0.5 * _safe_float(metrics.get("drawdown"))
        - 0.5 * _safe_float(metrics.get("turnover"))
        - 0.5 * _safe_float(metrics.get("vol_tracking_error"))
        - 1.0 * _safe_float(metrics.get("qlike"))
    )


def _as_lower_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).lower() for item in value]
    return [str(value).lower()]


def _pair_allowed_by_router_params(pair: Any, params: dict[str, Any]) -> bool:
    excluded_estimators = _as_lower_list(params.get("excluded_estimators", []))
    excluded_controllers = _as_lower_list(params.get("excluded_controllers", []))
    excluded_pairs_containing = _as_lower_list(params.get("excluded_pairs_containing", []))
    metadata = pair.metadata if isinstance(pair.metadata, dict) else {}
    estimator_text = " ".join([pair.estimator_name, str(metadata.get("estimator_path", ""))]).lower()
    controller_text = " ".join([pair.controller_name, str(metadata.get("controller_path", ""))]).lower()
    pair_text = " ".join([pair.name, estimator_text, controller_text]).lower()
    if any(token in estimator_text for token in excluded_estimators):
        return False
    if any(token in controller_text for token in excluded_controllers):
        return False
    if any(token in pair_text for token in excluded_pairs_containing):
        return False
    return True


def _deterministic_pair_ranking(
    train_pair_metrics: dict[str, dict[str, float]],
    best_pair: str,
) -> list[dict[str, Any]]:
    rows = []
    for pair_name, metrics in train_pair_metrics.items():
        rows.append(
            {
                "pair": pair_name,
                "turnover": _safe_float(metrics.get("turnover")),
                "drawdown": _safe_float(metrics.get("drawdown")),
                "sharpe": _safe_float(metrics.get("cost_adjusted_sharpe")),
                "vol_tracking_error": _safe_float(metrics.get("vol_tracking_error")),
                "qlike": _safe_float(metrics.get("qlike")),
                "deterministic_score": _metrics_to_score(metrics),
                "is_train_score_best": pair_name == best_pair,
            }
        )
    rows.sort(
        key=lambda row: (
            row["turnover"],
            row["drawdown"],
            -row["sharpe"],
            row["vol_tracking_error"],
            row["qlike"],
            -row["deterministic_score"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["is_rank_best"] = rank == 1
    return rows


def _pair_window_metric(pair_df: pd.DataFrame, target_vol: float) -> dict[str, float]:
    if pair_df.empty:
        return {}
    returns = pd.to_numeric(pair_df["returns"], errors="coerce").fillna(0.0)
    weight = pd.to_numeric(pair_df.get("weight", pd.Series(0.0, index=pair_df.index)), errors="coerce").fillna(0.0)
    turnover = weight.diff().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    realized_vol = _ann_vol(returns)
    equity = np.exp(returns.cumsum()).to_numpy()
    return {
        "obs": int(len(returns)),
        "rolling_sharpe": _rolling_sharpe(returns.to_numpy()),
        "drawdown": _max_drawdown(equity),
        "realized_vol": realized_vol,
        "vol_tracking_error": abs(realized_vol - target_vol),
        "turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "trailing_return": float(returns.sum()),
        "annualized_return": float(returns.mean() * 252.0) if len(returns) else 0.0,
        "cvar_95": _cvar_95(returns.to_numpy()),
    }


def _recent_pair_rankings(
    pair_results: dict[str, pd.DataFrame],
    eligible_pair_names: set[str],
    end_pos: int,
    target_vol: float,
    windows: tuple[int, ...] = (100, 60, 20),
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for window in windows:
        metrics_by_pair: dict[str, dict[str, float]] = {}
        rows: list[dict[str, Any]] = []
        for pair_name, pair_df in pair_results.items():
            if pair_name not in eligible_pair_names:
                continue
            hist = pair_df.iloc[max(0, end_pos - int(window)) : end_pos].copy()
            metrics = _pair_window_metric(hist, target_vol=target_vol)
            if not metrics or int(metrics.get("obs", 0)) < max(5, min(int(window), 20)):
                continue
            metrics_by_pair[pair_name] = metrics
            rows.append(
                {
                    "pair": pair_name,
                    "obs": int(metrics["obs"]),
                    "turnover": _safe_float(metrics.get("turnover")),
                    "drawdown": _safe_float(metrics.get("drawdown")),
                    "sharpe": _safe_float(metrics.get("rolling_sharpe")),
                    "realized_vol": _safe_float(metrics.get("realized_vol")),
                    "vol_tracking_error": _safe_float(metrics.get("vol_tracking_error")),
                    "trailing_return": _safe_float(metrics.get("trailing_return")),
                    "annualized_return": _safe_float(metrics.get("annualized_return")),
                    "cvar_95": _safe_float(metrics.get("cvar_95")),
                }
            )
        rows.sort(
            key=lambda row: (
                row["turnover"],
                row["drawdown"],
                -row["sharpe"],
                row["vol_tracking_error"],
                -row["annualized_return"],
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        out[f"{int(window)}d"] = {
            "window_days": int(window),
            "rank_rule": "turnover ascending, then drawdown ascending, then Sharpe descending, using rows before trade date",
            "ranks": rows,
            "metrics_by_pair": metrics_by_pair,
        }
    return out


def evaluate_precomputed_router_protocol(
    config_path: str,
    pair_results_dir: Path,
    train_days: int,
    test_days: int,
    step_days: int,
    max_pairs: int | None,
    output_dir: Path,
    config: dict | None = None,
    verbose: bool = True,
    oos_start_date: str | None = None,
    oos_end_date: str | None = None,
    single_oos_window: bool = False,
    freeze_oos_metrics: bool = False,
) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    if config is None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config

    market_df = _load_data(cfg)
    pairs = _build_pairs(cfg, max_pairs=max_pairs)
    pair_results = _load_pair_results(cfg, pair_results_dir, max_pairs=max_pairs)
    pairs = [pair for pair in pairs if pair.name in pair_results]
    if len(pairs) < 2:
        raise ValueError("Need at least 2 pair result files for protocol evaluation.")

    requested_oos_end = pd.Timestamp(oos_end_date) if oos_end_date else None
    if requested_oos_end is not None:
        cover_names = {
            name
            for name, pair_df in pair_results.items()
            if not pair_df.empty and pair_df.index.max() >= requested_oos_end
        }
        dropped_names = sorted(set(pair_results) - cover_names)
        if dropped_names and verbose:
            print(
                "[precomputed-router] "
                f"dropping {len(dropped_names)} pair result files that end before "
                f"{requested_oos_end.date()}: {dropped_names[:10]}"
                f"{'...' if len(dropped_names) > 10 else ''}",
                flush=True,
            )
        pair_results = {name: df for name, df in pair_results.items() if name in cover_names}
        pairs = [pair for pair in pairs if pair.name in pair_results]
        if len(pairs) < 2:
            raise ValueError(
                f"Need at least 2 pair result files covering requested OOS end {requested_oos_end.date()}."
            )

    common_index = market_df.index
    for pair_df in pair_results.values():
        common_index = common_index.intersection(pair_df.index)
    common_index = common_index.sort_values()
    market_df = market_df.loc[common_index]
    pair_results = {name: df.loc[common_index] for name, df in pair_results.items() if name in {p.name for p in pairs}}
    router_params = (cfg.get("router", {}).get("params", {}) or {})
    disable_regime_context = bool(router_params.get("disable_regime_context", False))
    disable_recent_rank_context = bool(router_params.get("disable_recent_rank_context", False))
    disable_benchmark_context = bool(router_params.get("disable_benchmark_context", False))
    disable_deterministic_baseline_context = bool(
        router_params.get("disable_deterministic_baseline_context", False)
    )

    benchmark_df = None
    benchmark_path = ((cfg.get("pair_history_features", {}) or {}).get("benchmark_path"))
    if benchmark_path and not disable_benchmark_context:
        path = Path(str(benchmark_path))
        if path.exists():
            benchmark_df = _normalize_pair_result(pd.read_parquet(path))
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
            benchmark_df = benchmark_df.sort_index().reindex(common_index)

    roll_window = int(cfg.get("roll_window", 252))
    target_vol = float(cfg.get("target_vol", 0.10))
    if disable_regime_context:
        market_regimes = pd.Series("middle", index=market_df.index, dtype="object")
    else:
        market_regimes = _historical_market_regimes(market_df, roll_window=roll_window)

    est_rows: list[dict[str, Any]] = []
    ctrl_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    oos_returns_all: list[float] = []
    oos_bh_returns_all: list[float] = []
    oos_ts_rows: list[dict[str, Any]] = []
    global_ai_call_cap = int((cfg.get("router", {}).get("params", {}) or {}).get("max_total_ai_calls", 0) or 0)
    global_ai_calls_used = 0

    start = roll_window + train_days
    if oos_start_date:
        requested_start = pd.Timestamp(oos_start_date)
        start = int(market_df.index.searchsorted(requested_start))
        start = max(start, roll_window + train_days)
    split_id = 0
    while start < len(market_df):
        split_t0 = time.time()
        train_start = start - train_days
        train_end = start
        test_end = len(market_df) if single_oos_window else min(start + test_days, len(market_df))
        if requested_oos_end is not None:
            test_end = min(test_end, int(market_df.index.searchsorted(requested_oos_end, side="right")))
        if test_end <= start:
            break

        train_pair_metrics: dict[str, dict[str, float]] = {}
        for pair in pairs:
            train_df = pair_results[pair.name].iloc[train_start:train_end]
            metrics = _pair_train_metrics(train_df, target_vol=target_vol)
            train_pair_metrics[pair.name] = metrics
            est_rows.append(
                {
                    "split": split_id,
                    "window": "train",
                    "pair": pair.name,
                    "qlike": metrics["qlike"],
                    "rmse_vol": metrics["rmse_vol"],
                    "turning_point_delay": metrics["turning_point_delay"],
                    "kupiec_lr": metrics["kupiec_lr"],
                    "christoffersen_lr": metrics["christoffersen_lr"],
                    "violation_rate": metrics["violation_rate"],
                    "qlike_high": metrics["qlike_high"],
                    "qlike_low": metrics["qlike_low"],
                }
            )
            ctrl_rows.append(
                {
                    "split": split_id,
                    "window": "train",
                    "pair": pair.name,
                    "vol_tracking_error": metrics["vol_tracking_error"],
                    "drawdown": metrics["drawdown"],
                    "tail_loss": metrics["tail_loss"],
                    "turnover": metrics["turnover"],
                    "cost_adjusted_sharpe": metrics["cost_adjusted_sharpe"],
                    "exposure_convexity": metrics["exposure_convexity"],
                }
            )

        params = dict(cfg.get("router", {}).get("params", {}) or {})
        eligible_pair_names = {
            pair.name for pair in pairs if _pair_allowed_by_router_params(pair, params)
        }
        eligible_train_pair_metrics = {
            name: metrics
            for name, metrics in train_pair_metrics.items()
            if name in eligible_pair_names
        }
        if not eligible_train_pair_metrics:
            eligible_train_pair_metrics = train_pair_metrics
        best_pair = max(eligible_train_pair_metrics.items(), key=lambda kv: _metrics_to_score(kv[1]))[0]
        deterministic_pair_ranking = (
            []
            if disable_deterministic_baseline_context
            else _deterministic_pair_ranking(eligible_train_pair_metrics, best_pair)
        )
        params["default_pair"] = best_pair
        if global_ai_call_cap > 0:
            remaining_ai_calls = max(global_ai_call_cap - global_ai_calls_used, 0)
            params["max_total_ai_calls"] = remaining_ai_calls
            params["ai_enabled"] = remaining_ai_calls > 0

        router = _build_router(cfg, pairs, params=params)
        pair_hist: dict[str, dict[str, deque]] = {}
        for pair in pairs:
            tm = train_pair_metrics[pair.name]
            pair_hist[pair.name] = {
                "qlike": deque([_safe_float(tm.get("qlike"))], maxlen=63),
                "turnover": deque([_safe_float(tm.get("turnover"))], maxlen=63),
                "vol_track": deque([_safe_float(tm.get("vol_tracking_error"))], maxlen=63),
                "returns": deque(maxlen=63),
                "equity": deque([1.0], maxlen=64),
            }

        n = test_end - start
        strategy_ret = np.zeros(n, dtype=float)
        equity = np.ones(n, dtype=float)
        bh_ret = np.zeros(n, dtype=float)
        bh_equity = np.ones(n, dtype=float)
        selected: list[str] = []
        selected_weights: list[float] = []
        selected_turnover: list[float] = []
        switched = 0
        prev_pair = None
        prev_weight = 0.0

        for j in range(n):
            i = start + j
            t = market_df.index[i]

            diagnostics: dict[str, dict[str, float]] = {}
            pair_perf: dict[str, dict[str, float]] = {}
            for pair in pairs:
                h = pair_hist[pair.name]
                returns = np.asarray(h["returns"], dtype=float)
                hist_equity = np.asarray(h["equity"], dtype=float)
                vol_track = float(np.mean(h["vol_track"])) if len(h["vol_track"]) else 0.0
                diagnostics[pair.name] = {
                    "estimator_loss": float(np.mean(h["qlike"])) if len(h["qlike"]) else 0.0,
                    "turnover": float(np.mean(h["turnover"])) if len(h["turnover"]) else 0.0,
                    "vol_tracking_error": vol_track,
                    "obs": int(len(returns)),
                }
                if len(returns) >= 2 and not freeze_oos_metrics:
                    pair_perf[pair.name] = {
                        "obs": int(len(returns)),
                        "rolling_sharpe": _rolling_sharpe(returns),
                        "drawdown": _max_drawdown(hist_equity),
                        "realized_vol": _ann_vol(returns),
                        "vol_tracking_error": vol_track,
                        "turnover": float(np.mean(h["turnover"])) if len(h["turnover"]) else 0.0,
                        "cvar_95": _cvar_95(returns),
                    }
                else:
                    tm = train_pair_metrics[pair.name]
                    pair_perf[pair.name] = {
                        "obs": int(train_days),
                        "rolling_sharpe": _safe_float(tm.get("cost_adjusted_sharpe")),
                        "drawdown": _safe_float(tm.get("drawdown")),
                        "realized_vol": _safe_float(tm.get("realized_vol"), target_vol),
                        "vol_tracking_error": _safe_float(tm.get("vol_tracking_error")),
                        "turnover": _safe_float(tm.get("turnover")),
                        "cvar_95": _safe_float(tm.get("tail_loss")),
                    }

            obs = strategy_ret[:j]
            realized_vol = _ann_vol(obs) if len(obs) >= 2 else 0.0
            history_end_pos = start if freeze_oos_metrics else i
            router_step = int(getattr(router, "_step", j))
            last_selection_step = getattr(router, "_last_ai_selection_step", None)
            selection_interval = int(getattr(router, "ai_selection_interval", 1))
            needs_selection_context = (
                last_selection_step is None
                or router_step - int(last_selection_step) >= selection_interval
            )
            perf = {
                "obs": int(j),
                "rolling_sharpe": _rolling_sharpe(obs),
                "drawdown": _max_drawdown(equity[: max(j, 1)]),
                "realized_vol": realized_vol,
                "vol_tracking_error": abs(realized_vol - target_vol) if j >= 2 else 0.0,
                "benchmark": {}
                if disable_benchmark_context
                else _benchmark_metrics(benchmark_df, end_pos=history_end_pos, target_vol=target_vol),
                "deterministic_best_pair": best_pair,
                "deterministic_pair_ranking": deterministic_pair_ranking,
                "deterministic_pair_rank_rule": (
                    "rank pairs by turnover ascending, then drawdown ascending, then Sharpe descending; "
                    "is_train_score_best marks the train-window champion chosen by the fallback score"
                ),
                "regime_pair_history": _pair_regime_history_metrics(
                    pair_results=pair_results,
                    market_regimes=market_regimes,
                    end_pos=history_end_pos,
                    target_vol=target_vol,
                )
                if needs_selection_context and not disable_regime_context
                else {},
                "recent_pair_rankings": _recent_pair_rankings(
                    pair_results=pair_results,
                    eligible_pair_names=eligible_pair_names,
                    end_pos=history_end_pos,
                    target_vol=target_vol,
                )
                if needs_selection_context and not disable_recent_rank_context
                else {},
                **pair_perf,
            }
            features = _build_market_features(market_df["returns_clean"].iloc[i - roll_window : i])

            pair = router.select(features, diagnostics, perf, timestamp=t)
            if verbose and needs_selection_context:
                decision = getattr(router, "decisions", [{}])[-1] if getattr(router, "decisions", []) else {}
                ai_response = decision.get("ai_selection_response", {}) if isinstance(decision, dict) else {}
                switch_decision = ai_response.get("decision", {}) if isinstance(ai_response, dict) else {}
                selection_response = ai_response.get("selection", {}) if isinstance(ai_response, dict) else {}
                switch_action = str(switch_decision.get("action", "n/a"))
                selected_by_ai = bool(selection_response)
                print(
                    f"[decision {split_id:03d}:{j:04d}] {t.date()} | "
                    f"regime={features.get('vol_regime', 'n/a')} | "
                    f"ai_regime={decision.get('market_features', {}).get('vol_regime', 'n/a')} | "
                    f"switch_check={switch_action} | "
                    f"selection_call={'yes' if selected_by_ai else 'no'} | "
                    f"active={pair.name} | "
                    f"switched={decision.get('switched', False)} | "
                    f"ai_calls={decision.get('ai_selection_call_count', 'n/a')} | "
                    f"error={decision.get('ai_selection_error', '') or '-'}",
                    flush=True,
                )
            selected.append(pair.name)
            if prev_pair is not None and pair.name != prev_pair:
                switched += 1
            prev_pair = pair.name

            pair_df = pair_results[pair.name]
            r = _safe_float(pair_df["returns"].iloc[i])
            b = _safe_float(pair_df["asset_returns"].iloc[i])
            w = _safe_float(pair_df.get("weight", pd.Series(0.0, index=pair_df.index)).iloc[i])
            selected_weights.append(w)
            selected_turnover.append(abs(w - prev_weight))
            prev_weight = w
            strategy_ret[j] = r
            bh_ret[j] = b
            equity[j] = (equity[j - 1] if j else 1.0) * math.exp(r)
            bh_equity[j] = (bh_equity[j - 1] if j else 1.0) * math.exp(b)

            vol_est = _safe_float(pair_df.get("vol_estimate", pd.Series(np.nan, index=pair_df.index)).iloc[i], np.nan)
            if np.isfinite(vol_est) and vol_est > 0:
                rv2 = 252.0 * (b**2)
                pair_hist[pair.name]["qlike"].append(float(math.log(vol_est**2) + rv2 / (vol_est**2)))
            weight = pair_df.get("weight", pd.Series(0.0, index=pair_df.index)).astype(float)
            if i > 0:
                pair_hist[pair.name]["turnover"].append(float(abs(_safe_float(weight.iloc[i]) - _safe_float(weight.iloc[i - 1]))))
            pair_hist[pair.name]["vol_track"].append(abs(_ann_vol(strategy_ret[: j + 1]) - target_vol) if j >= 2 else 0.0)
            pair_hist[pair.name]["returns"].append(float(r))
            pair_hist[pair.name]["equity"].append(float(equity[j]))

        oos_returns_all.extend(strategy_ret.tolist())
        oos_bh_returns_all.extend(bh_ret.tolist())
        split_dates = market_df.index[start:test_end]
        for k in range(n):
            oos_ts_rows.append(
                {
                    "split": split_id,
                    "date": split_dates[k],
                    "router_return": strategy_ret[k],
                    "bh_return": bh_ret[k],
                    "router_equity": equity[k],
                    "bh_equity": bh_equity[k],
                    "selected_pair": selected[k],
                    "selected_weight": selected_weights[k],
                    "selected_turnover": selected_turnover[k],
                }
            )

        market_vol = pd.Series(market_df["returns_clean"].iloc[start:test_end].to_numpy()).rolling(21).std() * math.sqrt(252.0)
        med_vol = float(np.nanmedian(market_vol)) if np.isfinite(market_vol).any() else np.nan
        high_mask = market_vol >= med_vol if np.isfinite(med_vol) else pd.Series([False] * len(market_vol))
        rr = pd.Series(strategy_ret)
        router_decisions = list(getattr(router, "decisions", []))
        ai_regime_errors = [
            str(decision.get("market_features", {}).get("ai_regime_error", ""))
            for decision in router_decisions
            if str(decision.get("market_features", {}).get("ai_regime_error", ""))
        ]
        ai_selection_errors = [
            str(decision.get("ai_selection_error", ""))
            for decision in router_decisions
            if str(decision.get("ai_selection_error", ""))
        ]

        router_rows.append(
            {
                "split": split_id,
                "window": "oos",
                "start": str(market_df.index[start].date()),
                "end": str(market_df.index[test_end - 1].date()),
                "sharpe": _rolling_sharpe(strategy_ret),
                "calmar": _calmar(strategy_ret),
                "cvar_95": _cvar_95(strategy_ret),
                "bh_sharpe": _rolling_sharpe(bh_ret),
                "bh_calmar": _calmar(bh_ret),
                "bh_cvar_95": _cvar_95(bh_ret),
                "excess_sharpe": _rolling_sharpe(strategy_ret) - _rolling_sharpe(bh_ret),
                "regime_sharpe_high_vol": _rolling_sharpe(rr[high_mask.fillna(False)].to_numpy()) if high_mask.any() else np.nan,
                "regime_sharpe_low_vol": _rolling_sharpe(rr[~high_mask.fillna(False)].to_numpy()) if (~high_mask.fillna(False)).any() else np.nan,
                "switch_count": switched,
                "switch_rate": switched / max(len(selected), 1),
                "turnover": float(np.mean(selected_turnover)) if selected_turnover else 0.0,
                "drawdown": _max_drawdown(equity),
                "bh_drawdown": _max_drawdown(bh_equity),
                "dd_behavior_proxy": np.nan,
                "best_train_pair": best_pair,
                "router_type": str(cfg.get("router", {}).get("type", "base")),
                "unique_selected": int(pd.Series(selected).nunique()),
                "ai_regime_call_count": int(getattr(router, "_ai_regime_call_count", 0)),
                "ai_selection_call_count": int(getattr(router, "_ai_selection_call_count", 0)),
                "ai_total_call_count": int(
                    getattr(router, "_ai_regime_call_count", 0)
                    + getattr(router, "_ai_selection_call_count", 0)
                ),
                "ai_regime_failure_count": len(ai_regime_errors),
                "ai_selection_failure_count": len(ai_selection_errors),
                "ai_failure_count": len(ai_regime_errors) + len(ai_selection_errors),
                "ai_regime_failure_sample": " | ".join(ai_regime_errors[:3]),
                "ai_selection_failure_sample": " | ".join(ai_selection_errors[:3]),
                "global_ai_calls_used_before_split": int(global_ai_calls_used),
            }
        )
        global_ai_calls_used += int(
            getattr(router, "_ai_regime_call_count", 0)
            + getattr(router, "_ai_selection_call_count", 0)
        )

        if verbose:
            print(
                f"[split {split_id:03d}] "
                f"{market_df.index[start].date()} -> {market_df.index[test_end - 1].date()} | "
                f"best_pair={best_pair} | unique={pd.Series(selected).nunique()} | "
                f"split_time={time.time() - split_t0:.1f}s | total_time={time.time() - t0:.1f}s"
            )

        start += step_days
        split_id += 1
        if single_oos_window:
            break

    oos = np.asarray(oos_returns_all, dtype=float)
    oos_bh = np.asarray(oos_bh_returns_all, dtype=float)
    summary = pd.DataFrame(
        [
            {
                "overall_sharpe": _rolling_sharpe(oos),
                "overall_calmar": _calmar(oos),
                "overall_cvar_95": _cvar_95(oos),
                "overall_ann_return": float(np.mean(oos) * 252.0),
                "overall_ann_vol": _ann_vol(oos),
                "overall_drawdown": _max_drawdown(np.exp(np.cumsum(oos))),
                "bh_overall_sharpe": _rolling_sharpe(oos_bh),
                "bh_overall_calmar": _calmar(oos_bh),
                "bh_overall_cvar_95": _cvar_95(oos_bh),
                "bh_ann_return": float(np.mean(oos_bh) * 252.0),
                "bh_ann_vol": _ann_vol(oos_bh),
                "bh_drawdown": _max_drawdown(np.exp(np.cumsum(oos_bh))),
                "excess_sharpe": _rolling_sharpe(oos) - _rolling_sharpe(oos_bh),
                "n_oos_obs": int(len(oos)),
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    est_df = pd.DataFrame(est_rows)
    ctrl_df = pd.DataFrame(ctrl_rows)
    router_df = pd.DataFrame(router_rows)
    ts_df = pd.DataFrame(oos_ts_rows)

    est_df.to_csv(output_dir / "estimator_diagnostics.csv", index=False)
    ctrl_df.to_csv(output_dir / "controller_diagnostics.csv", index=False)
    router_df.to_csv(output_dir / "router_oos_diagnostics.csv", index=False)
    ts_df.to_csv(output_dir / "router_vs_bh_timeseries.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    return {"estimator": est_df, "controller": ctrl_df, "router": router_df, "timeseries": ts_df, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Evaluate precomputed router walk-forward protocol")
    parser.add_argument("--strategy", "-s", default="configs/strategies/router_master.yaml")
    parser.add_argument("--pair-results-dir", default="results/all_estimator_controller_pairs")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--oos-start-date", default=None)
    parser.add_argument("--oos-end-date", default=None)
    parser.add_argument("--single-oos-window", action="store_true")
    parser.add_argument("--freeze-oos-metrics", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-dir", default=str(Env.path("evaluation") / "router_protocol_precomputed"))
    args = parser.parse_args()

    tables = evaluate_precomputed_router_protocol(
        config_path=args.strategy,
        pair_results_dir=Path(args.pair_results_dir),
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        max_pairs=args.max_pairs,
        output_dir=Path(args.output_dir),
        oos_start_date=args.oos_start_date,
        oos_end_date=args.oos_end_date,
        single_oos_window=args.single_oos_window,
        freeze_oos_metrics=args.freeze_oos_metrics,
    )

    print("=" * 60)
    print("Precomputed router protocol evaluation finished")
    print(f"Output dir: {args.output_dir}")
    print(tables["summary"].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
