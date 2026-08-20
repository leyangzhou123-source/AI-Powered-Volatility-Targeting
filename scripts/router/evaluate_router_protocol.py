"""Evaluate router with walk-forward nested protocol.

Protocol implemented:
1) Outer loop: walk-forward train/test splits (OOS evaluation).
2) Inner loop: evaluate each estimator-controller pair on train split.
3) OOS only: configured router selects pair online and updates diagnostics.
4) Exports estimator/controller/router diagnostics and OOS summary.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.env import Env
from src.router import (
    BaseRuleBasedRouter,
    ContextualBanditRouter,
    MixtureOfExpertsRouter,
    Router,
    RuleConstraintRouter,
    StrategyPair,
)


def _load_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_rebalance_dates(dates: pd.DatetimeIndex, rebalance_freq: str) -> set[pd.Timestamp]:
    if rebalance_freq is None:
        rebalance_freq = "D"
    f = str(rebalance_freq).upper()
    if f in ("D", "DAILY", "1D"):
        return set(dates)
    schedule = pd.date_range(dates.min(), dates.max(), freq=f)
    return set(dates.intersection(schedule))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _rolling_sharpe(returns: np.ndarray, ann: float = 252.0) -> float:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float((r.mean() / sd) * math.sqrt(ann))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    e = pd.Series(equity_curve).replace([np.inf, -np.inf], np.nan).dropna()
    if len(e) < 2:
        return 0.0
    dd = e / e.cummax() - 1.0
    return float(-dd.min())


def _cvar_95(returns: np.ndarray) -> float:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 10:
        return 0.0
    q = float(np.quantile(r, 0.05))
    tail = r[r <= q]
    if len(tail) == 0:
        return 0.0
    return float(-tail.mean())


def _calmar(returns: np.ndarray, ann: float = 252.0) -> float:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return 0.0
    ann_ret = float(r.mean() * ann)
    mdd = _max_drawdown(np.exp(np.cumsum(r.to_numpy())))
    if mdd <= 0:
        return 0.0
    return float(ann_ret / mdd)


def _qlike_var(rv2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-12) -> float:
    s2 = np.maximum(sigma2, eps)
    return float(np.mean(np.log(s2) + rv2 / s2))


def _turning_point_delay(pred_vol: pd.Series, real_vol: pd.Series, max_lag: int = 10) -> float:
    p = pd.Series(pred_vol).astype(float)
    r = pd.Series(real_vol).astype(float)
    if len(p) < max_lag + 5 or len(r) < max_lag + 5:
        return np.nan

    dp = np.sign(p.diff()).fillna(0.0)
    dr = np.sign(r.diff()).fillna(0.0)

    best_lag = np.nan
    best_corr = -np.inf
    for lag in range(max_lag + 1):
        y = dr.shift(-lag)
        tmp = pd.concat([dp.rename("x"), y.rename("y")], axis=1).dropna()
        if len(tmp) < 5:
            continue
        # Avoid numpy divide-by-zero warnings when one side is constant.
        if float(tmp["x"].std(ddof=1)) <= 0 or float(tmp["y"].std(ddof=1)) <= 0:
            c = -np.inf
        else:
            c = tmp["x"].corr(tmp["y"])
            c = -np.inf if pd.isna(c) else float(c)
        if c > best_corr:
            best_corr = c
            best_lag = float(lag)
    return best_lag


def _kupiec_test(violations: np.ndarray, alpha: float = 0.95) -> dict[str, float]:
    v = np.asarray(violations, dtype=int)
    n = len(v)
    if n == 0:
        return {"kupiec_lr": np.nan, "violation_rate": np.nan}
    x = int(v.sum())
    p = 1.0 - alpha
    phat = x / max(n, 1)
    if phat in (0.0, 1.0):
        return {"kupiec_lr": np.nan, "violation_rate": phat}
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    return {"kupiec_lr": float(-2.0 * (ll0 - ll1)), "violation_rate": phat}


def _christoffersen_independence(violations: np.ndarray) -> dict[str, float]:
    v = np.asarray(violations, dtype=int)
    if len(v) < 3:
        return {"christoffersen_lr": np.nan}
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(v)):
        prev, curr = v[i - 1], v[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1
    d0 = n00 + n01
    d1 = n10 + n11
    if d0 == 0 or d1 == 0:
        return {"christoffersen_lr": np.nan}
    p01 = n01 / d0
    p11 = n11 / d1
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if p in (0.0, 1.0) or p01 in (0.0, 1.0) or p11 in (0.0, 1.0):
        return {"christoffersen_lr": np.nan}
    ll_ind = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    ll_dep = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)
    return {"christoffersen_lr": float(-2.0 * (ll_ind - ll_dep))}


@dataclass
class PairRun:
    returns: np.ndarray
    vol_forecasts: np.ndarray
    weights: np.ndarray
    turnover: np.ndarray
    equity: np.ndarray


class PairRunner:
    """Runs one estimator-controller pair on a date slice."""

    def __init__(
        self,
        target_vol: float,
        cost_bps: float,
        w_min: float,
        w_max: float,
        roll_window: int,
        rebalance_freq: str,
    ):
        self.target_vol = target_vol
        self.cost_bps = cost_bps
        self.w_min = w_min
        self.w_max = w_max
        self.roll_window = roll_window
        self.rebalance_freq = rebalance_freq

    def run(
        self,
        pair: StrategyPair,
        df_all: pd.DataFrame,
        start_pos: int,
        end_pos: int,
    ) -> PairRun:
        dates_all = df_all.index
        rebal_dates = _get_rebalance_dates(dates_all[start_pos:end_pos], self.rebalance_freq)

        n = max(0, end_pos - start_pos)
        if n <= 1:
            return PairRun(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

        strat_log_r = np.zeros(n, dtype=float)
        vol_fc = np.full(n, np.nan, dtype=float)
        weights = np.zeros(n, dtype=float)
        turnover = np.zeros(n, dtype=float)
        equity = np.ones(n, dtype=float)

        prev_w = 0.0

        for j in range(n - 1):
            i = start_pos + j
            t = dates_all[i]

            window = df_all["returns_clean"].iloc[i - self.roll_window : i].dropna().astype(float)
            if len(window) == 0:
                v = np.nan
            else:
                try:
                    if hasattr(pair.estimator, "estimate_window"):
                        v = float(pair.estimator.estimate_window(window))
                    else:
                        v = float(pair.estimator.estimate(t, window))
                except Exception:
                    v = np.nan
            vol_fc[j] = v

            if hasattr(pair.controller, "update"):
                pair.controller.update(vol_estimate=v, ret=float(df_all["returns_clean"].iloc[i]), equity=float(equity[j]))

            if t in rebal_dates:
                try:
                    w = float(pair.controller.compute_weight(self.target_vol, v, prev_w))
                except Exception:
                    w = float(prev_w)
                w = float(np.clip(w, self.w_min, self.w_max))
            else:
                w = float(prev_w)

            weights[j] = w
            turnover[j] = abs(w - prev_w)
            cost = turnover[j] * (self.cost_bps / 10000.0)
            next_r = float(df_all["returns"].iloc[i + 1])
            strat_log_r[j + 1] = w * next_r - cost
            prev_w = w
            equity[j + 1] = equity[j] * math.exp(strat_log_r[j + 1])

        weights[-1] = prev_w
        vol_fc[-1] = vol_fc[-2] if n > 1 else np.nan
        turnover[-1] = 0.0

        return PairRun(strat_log_r, vol_fc, weights, turnover, equity)


def _estimator_diagnostics(run: PairRun, raw_returns: pd.Series, ann: float = 252.0) -> dict[str, float]:
    n = len(run.vol_forecasts)
    if n < 5:
        return {
            "qlike": np.nan,
            "rmse_vol": np.nan,
            "turning_point_delay": np.nan,
            "kupiec_lr": np.nan,
            "christoffersen_lr": np.nan,
            "violation_rate": np.nan,
            "qlike_high": np.nan,
            "qlike_low": np.nan,
        }

    sigma = pd.Series(run.vol_forecasts)
    rv_next = np.sqrt(np.maximum(ann * raw_returns.shift(-1).to_numpy()[:n] ** 2, 1e-12))
    rv_next = pd.Series(rv_next)

    valid = sigma.notna() & rv_next.notna() & (sigma > 0) & (rv_next > 0)
    if valid.sum() < 5:
        return {
            "qlike": np.nan,
            "rmse_vol": np.nan,
            "turning_point_delay": np.nan,
            "kupiec_lr": np.nan,
            "christoffersen_lr": np.nan,
            "violation_rate": np.nan,
            "qlike_high": np.nan,
            "qlike_low": np.nan,
        }

    s = sigma[valid].to_numpy()
    r = rv_next[valid].to_numpy()
    qlike = _qlike_var(r**2, s**2)
    rmse_vol = float(np.sqrt(np.mean((s - r) ** 2)))
    tpd = _turning_point_delay(pd.Series(s), pd.Series(r))

    var95 = -1.645 * s / math.sqrt(ann)
    realized = raw_returns.to_numpy()[: len(s)]
    viol = (realized < var95).astype(int)

    kupiec = _kupiec_test(viol, alpha=0.95)
    ch = _christoffersen_independence(viol)

    mid = float(np.nanmedian(r))
    high_mask = r >= mid
    low_mask = r < mid
    qh = _qlike_var(r[high_mask] ** 2, s[high_mask] ** 2) if high_mask.sum() >= 3 else np.nan
    ql = _qlike_var(r[low_mask] ** 2, s[low_mask] ** 2) if low_mask.sum() >= 3 else np.nan

    return {
        "qlike": qlike,
        "rmse_vol": rmse_vol,
        "turning_point_delay": tpd,
        "kupiec_lr": float(kupiec["kupiec_lr"]),
        "christoffersen_lr": float(ch["christoffersen_lr"]),
        "violation_rate": float(kupiec["violation_rate"]),
        "qlike_high": qh,
        "qlike_low": ql,
    }


def _controller_diagnostics(run: PairRun, target_vol: float, ann: float = 252.0) -> dict[str, float]:
    r = run.returns
    if len(r) < 5:
        return {
            "vol_tracking_error": np.nan,
            "drawdown": np.nan,
            "tail_loss": np.nan,
            "turnover": np.nan,
            "cost_adjusted_sharpe": np.nan,
            "exposure_convexity": np.nan,
        }

    realized_vol = float(np.nanstd(r, ddof=1) * math.sqrt(ann))
    vol_err = abs(realized_vol - target_vol)
    dd = _max_drawdown(run.equity)
    tail = _cvar_95(r)
    to = float(np.nanmean(run.turnover))
    sharpe = _rolling_sharpe(r, ann=ann)

    # Optional convexity proxy: lower exposure on high-vol days.
    vol_spike = pd.Series(np.abs(r)).rolling(21).mean().to_numpy()
    if np.isfinite(vol_spike).sum() > 10:
        th = np.nanquantile(vol_spike, 0.8)
        mask = vol_spike >= th
        if mask.sum() > 3:
            convexity = float(np.nanmean(run.weights[mask]) - np.nanmean(run.weights[~mask]))
        else:
            convexity = np.nan
    else:
        convexity = np.nan

    return {
        "vol_tracking_error": vol_err,
        "drawdown": dd,
        "tail_loss": tail,
        "turnover": to,
        "cost_adjusted_sharpe": sharpe,
        "exposure_convexity": convexity,
    }


def _build_pairs(cfg: dict, max_pairs: int | None = None) -> list[StrategyPair]:
    router_cfg = cfg.get("router", {})
    pairs_cfg = router_cfg.get("pairs", [])
    out: list[StrategyPair] = []
    for i, p in enumerate(pairs_cfg):
        if max_pairs is not None and i >= max_pairs:
            break
        try:
            est_cfg = p["estimator"]
            ctrl_cfg = p["controller"]
            est_cls = _load_class(est_cfg["class"])
            ctrl_cls = _load_class(ctrl_cfg["class"])
            out.append(
                StrategyPair(
                    name=p.get("name", f"pair_{i}"),
                    estimator=est_cls(est_cfg.get("params", {})),
                    controller=ctrl_cls(ctrl_cfg.get("params", {})),
                    metadata={
                        "estimator_path": est_cfg["class"],
                        "controller_path": ctrl_cfg["class"],
                    },
                )
            )
        except Exception:
            continue
    return out


def _build_router(cfg: dict, pairs: list[StrategyPair], params: dict[str, Any]) -> Router:
    router_cfg = cfg.get("router", {}) or {}
    router_class_path = router_cfg.get("class")
    if router_class_path:
        router_cls = _load_class(str(router_class_path))
        return router_cls(pairs=pairs, params=params)

    router_type = str(router_cfg.get("type", "base")).lower()
    router_map = {
        "base": BaseRuleBasedRouter,
        "base_rule": BaseRuleBasedRouter,
        "base_rule_based": BaseRuleBasedRouter,
        "rule_based": BaseRuleBasedRouter,
        "rule_constraint": RuleConstraintRouter,
        "rule_constrained": RuleConstraintRouter,
        "constraint": RuleConstraintRouter,
        "contextual_bandit": ContextualBanditRouter,
        "bandit": ContextualBanditRouter,
        "linucb": ContextualBanditRouter,
        "moe": MixtureOfExpertsRouter,
        "mixture_of_experts": MixtureOfExpertsRouter,
    }
    router_cls = router_map.get(router_type, BaseRuleBasedRouter)
    return router_cls(pairs=pairs, params=params)


def _load_data(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("path")
    if data_path is None:
        data_path = Env.path("processed") / "ES_Daily_Processed.parquet"
    else:
        data_path = Path(data_path)
        if not data_path.is_absolute():
            data_path = Env.path("root") / data_path
    df = pd.read_parquet(data_path)
    date_col = data_cfg.get("date_col")
    if date_col is None:
        for candidate in ("date", "timestamp"):
            if candidate in df.columns:
                date_col = candidate
                break
    if date_col is not None and date_col in df.columns:
        df = df.set_index(pd.to_datetime(df[date_col]))
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    returns_col = data_cfg.get("returns_col")
    if returns_col is None:
        for candidate in ("returns", "returns_clean", "log_return", "simple_return"):
            if candidate in df.columns:
                returns_col = candidate
                break
    if returns_col and returns_col in df.columns and "returns" not in df.columns:
        df["returns"] = df[returns_col]
    if "returns" in df.columns and "returns_clean" not in df.columns:
        df["returns_clean"] = df["returns"]
    if "returns" not in df.columns or "returns_clean" not in df.columns:
        raise ValueError("Data must contain returns and returns_clean columns")
    df["returns"] = pd.to_numeric(df["returns"], errors="coerce")
    df["returns_clean"] = pd.to_numeric(df["returns_clean"], errors="coerce")
    return df


def _build_market_features(window: pd.Series) -> dict[str, Any]:
    r = pd.Series(window).dropna().astype(float)
    if len(r) < 2:
        return {"rolling_vol": np.nan, "rolling_mean": np.nan, "rolling_skew": np.nan, "vol_regime": "mid"}
    rv = float(r.std(ddof=1) * math.sqrt(252))
    mu = float(r.mean() * 252)
    sk = float(r.skew()) if np.isfinite(r.skew()) else 0.0
    regime = "high" if abs(r.iloc[-1]) > abs(r.quantile(0.8)) else "low" if abs(r.iloc[-1]) < abs(r.quantile(0.2)) else "mid"
    return {"rolling_vol": rv, "rolling_mean": mu, "rolling_skew": sk, "vol_regime": regime}


def evaluate_router_protocol(
    config_path: str,
    train_days: int,
    test_days: int,
    step_days: int,
    max_pairs: int | None,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = _load_data(cfg)
    roll_window = int(cfg.get("roll_window", 252))
    target_vol = float(cfg.get("target_vol", 0.10))
    cost_bps = float(cfg.get("cost_bps", 5.0))
    w_min = float(cfg.get("weight_min", 0.0))
    w_max = float(cfg.get("weight_max", 1.5))
    rebalance_freq = cfg.get("rebalance_freq", "daily")

    pairs = _build_pairs(cfg, max_pairs=max_pairs)
    if len(pairs) < 2:
        raise ValueError("Need at least 2 valid router pairs for protocol evaluation.")

    runner = PairRunner(target_vol, cost_bps, w_min, w_max, roll_window, rebalance_freq)

    est_rows: list[dict[str, Any]] = []
    ctrl_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    oos_returns_all: list[float] = []
    oos_bh_returns_all: list[float] = []
    oos_ts_rows: list[dict[str, Any]] = []

    start = roll_window + train_days
    split_id = 0

    while start + test_days < len(df):
        split_t0 = time.time()
        train_start = start - train_days
        train_end = start
        test_end = min(start + test_days, len(df))

        train_pair_metrics: dict[str, dict[str, float]] = {}
        score_rows = []

        # Nested inner selection on train window.
        for pair in pairs:
            pair.estimator = pair.estimator.__class__(getattr(pair.estimator, "params", {}))
            pair.controller = pair.controller.__class__(getattr(pair.controller, "params", {}))
            run = runner.run(pair, df, train_start, train_end)

            est_diag = _estimator_diagnostics(run, df["returns"].iloc[train_start:train_end])
            ctrl_diag = _controller_diagnostics(run, target_vol=target_vol)
            train_pair_metrics[pair.name] = {**est_diag, **ctrl_diag}

            score = (
                1.0 * _safe_float(ctrl_diag.get("cost_adjusted_sharpe"))
                - 0.5 * _safe_float(ctrl_diag.get("drawdown"))
                - 0.5 * _safe_float(ctrl_diag.get("turnover"))
                - 0.5 * _safe_float(ctrl_diag.get("vol_tracking_error"))
                - 1.0 * _safe_float(est_diag.get("qlike"))
            )
            score_rows.append((pair.name, score))

            est_rows.append({"split": split_id, "window": "train", "pair": pair.name, **est_diag})
            ctrl_rows.append({"split": split_id, "window": "train", "pair": pair.name, **ctrl_diag})

        score_rows.sort(key=lambda x: x[1], reverse=True)
        best_pair = score_rows[0][0]

        # Constraint thresholds estimated from train diagnostics.
        est_losses = [
            _safe_float(train_pair_metrics[p].get("qlike"), np.nan)
            for p in train_pair_metrics
            if np.isfinite(_safe_float(train_pair_metrics[p].get("qlike"), np.nan))
        ]
        turnovers = [_safe_float(train_pair_metrics[p].get("turnover"), 0.0) for p in train_pair_metrics]
        vol_errs = [_safe_float(train_pair_metrics[p].get("vol_tracking_error"), 0.0) for p in train_pair_metrics]

        params = cfg.get("router", {}).get("params", {}) or {}
        params = dict(params)
        params["default_pair"] = best_pair
        if est_losses:
            params.setdefault("max_estimator_loss", float(np.quantile(est_losses, 0.8)))
        params.setdefault("max_turnover", float(np.quantile(turnovers, 0.8)))
        params.setdefault("max_vol_tracking_error", float(np.quantile(vol_errs, 0.8)))

        # Re-instantiate for OOS run.
        oos_pairs = _build_pairs(cfg, max_pairs=max_pairs)
        router = _build_router(cfg, oos_pairs, params=params)

        n = test_end - start
        strategy_ret = np.zeros(n, dtype=float)
        equity = np.ones(n, dtype=float)
        bh_ret = np.zeros(n, dtype=float)
        bh_equity = np.ones(n, dtype=float)
        selected = []
        switched = 0
        prev_pair = None
        prev_w = 0.0

        pair_hist: dict[str, dict[str, deque]] = {}
        for p in oos_pairs:
            train_metrics = train_pair_metrics.get(p.name, {})
            pair_hist[p.name] = {
                "qlike": deque(maxlen=63),
                "turnover": deque(maxlen=63),
                "vol_track": deque(maxlen=63),
                "returns": deque(maxlen=63),
                "equity": deque(maxlen=64),
            }
            if np.isfinite(_safe_float(train_metrics.get("qlike"), np.nan)):
                pair_hist[p.name]["qlike"].append(_safe_float(train_metrics.get("qlike")))
            pair_hist[p.name]["turnover"].append(_safe_float(train_metrics.get("turnover")))
            pair_hist[p.name]["vol_track"].append(_safe_float(train_metrics.get("vol_tracking_error")))
            pair_hist[p.name]["equity"].append(1.0)

        rebal_dates = _get_rebalance_dates(df.index[start:test_end], rebalance_freq)

        for j in range(n - 1):
            i = start + j
            t = df.index[i]

            # Build per-pair diagnostics from OOS history only.
            diagnostics: dict[str, dict[str, float]] = {}
            pair_perf: dict[str, dict[str, float]] = {}
            for p in oos_pairs:
                h = pair_hist[p.name]
                qlike = float(np.mean(h["qlike"])) if len(h["qlike"]) else 0.0
                to = float(np.mean(h["turnover"])) if len(h["turnover"]) else 0.0
                ve = float(np.mean(h["vol_track"])) if len(h["vol_track"]) else 0.0
                pair_returns = np.asarray(h["returns"], dtype=float)
                pair_equity = np.asarray(h["equity"], dtype=float)
                diagnostics[p.name] = {
                    "estimator_loss": qlike,
                    "turnover": to,
                    "vol_tracking_error": ve,
                    "obs": len(pair_returns),
                }
                if len(pair_returns) >= 2:
                    pair_perf[p.name] = {
                        "obs": int(len(pair_returns)),
                        "rolling_sharpe": _rolling_sharpe(pair_returns),
                        "drawdown": _max_drawdown(pair_equity) if len(pair_equity) >= 2 else 0.0,
                        "realized_vol": float(np.std(pair_returns, ddof=1) * math.sqrt(252.0)),
                        "vol_tracking_error": ve,
                    }
                else:
                    tm = train_pair_metrics.get(p.name, {})
                    pair_perf[p.name] = {
                        "obs": int(train_days),
                        "rolling_sharpe": _safe_float(tm.get("cost_adjusted_sharpe")),
                        "drawdown": _safe_float(tm.get("drawdown")),
                        "realized_vol": max(target_vol - _safe_float(tm.get("vol_tracking_error")), 0.0),
                        "vol_tracking_error": _safe_float(tm.get("vol_tracking_error")),
                    }

            perf = {
                "obs": int(j),
                "rolling_sharpe": _rolling_sharpe(strategy_ret[:j]),
                "drawdown": _max_drawdown(equity[: max(j, 1)]),
                "realized_vol": float(np.std(strategy_ret[: max(j, 2)], ddof=1) * math.sqrt(252.0)) if j >= 2 else 0.0,
                "vol_tracking_error": abs(
                    float(np.std(strategy_ret[: max(j, 2)], ddof=1) * math.sqrt(252.0)) - target_vol
                )
                if j >= 2
                else 0.0,
                **pair_perf,
            }
            features = _build_market_features(df["returns_clean"].iloc[i - roll_window : i])

            pair = router.select(features, diagnostics, perf, timestamp=t)
            selected.append(pair.name)

            if prev_pair is not None and pair.name != prev_pair:
                switched += 1
            prev_pair = pair.name

            w_ret = df["returns_clean"].iloc[i - roll_window : i].dropna().astype(float)
            try:
                if hasattr(pair.estimator, "estimate_window"):
                    vol_est = float(pair.estimator.estimate_window(w_ret))
                else:
                    vol_est = float(pair.estimator.estimate(t, w_ret))
            except Exception:
                vol_est = np.nan

            if hasattr(pair.controller, "update"):
                pair.controller.update(vol_estimate=vol_est, ret=float(df["returns_clean"].iloc[i]), equity=float(equity[j]))

            if t in rebal_dates:
                try:
                    w = float(pair.controller.compute_weight(target_vol, vol_est, prev_w))
                except Exception:
                    w = float(prev_w)
                w = float(np.clip(w, w_min, w_max))
            else:
                w = float(prev_w)

            turnover = abs(w - prev_w)
            cost = turnover * (cost_bps / 10000.0)
            nxt = float(df["returns"].iloc[i + 1])
            strategy_ret[j + 1] = w * nxt - cost
            equity[j + 1] = equity[j] * math.exp(strategy_ret[j + 1])
            bh_ret[j + 1] = nxt
            bh_equity[j + 1] = bh_equity[j] * math.exp(bh_ret[j + 1])

            # Update selected-pair diagnostics with realized next-step info.
            rv2 = 252.0 * (nxt**2)
            sigma2 = max(vol_est**2 if np.isfinite(vol_est) else np.nan, 1e-12)
            if np.isfinite(sigma2):
                ql = math.log(sigma2) + rv2 / sigma2
                pair_hist[pair.name]["qlike"].append(float(ql))
            pair_hist[pair.name]["turnover"].append(float(turnover))
            pair_hist[pair.name]["vol_track"].append(abs(float(np.std(strategy_ret[: j + 2], ddof=1) * math.sqrt(252.0)) - target_vol) if j > 2 else 0.0)
            pair_hist[pair.name]["returns"].append(float(strategy_ret[j + 1]))
            pair_hist[pair.name]["equity"].append(float(equity[j + 1]))

            prev_w = w

        oos_returns_all.extend(strategy_ret.tolist())
        oos_bh_returns_all.extend(bh_ret.tolist())
        split_dates = df.index[start:test_end]
        for k in range(n):
            oos_ts_rows.append(
                {
                    "split": split_id,
                    "date": split_dates[k],
                    "router_return": strategy_ret[k],
                    "bh_return": bh_ret[k],
                    "router_equity": equity[k],
                    "bh_equity": bh_equity[k],
                }
            )

        market_vol = pd.Series(df["returns_clean"].iloc[start:test_end].to_numpy()).rolling(21).std() * math.sqrt(252.0)
        med_vol = float(np.nanmedian(market_vol)) if np.isfinite(market_vol).any() else np.nan
        high_mask = market_vol >= med_vol if np.isfinite(med_vol) else pd.Series([False] * len(market_vol))

        rr = pd.Series(strategy_ret)
        perf_high = _rolling_sharpe(rr[high_mask.fillna(False)].to_numpy()) if high_mask.any() else np.nan
        perf_low = _rolling_sharpe(rr[~high_mask.fillna(False)].to_numpy()) if (~high_mask.fillna(False)).any() else np.nan

        dd_series = pd.Series(equity) / pd.Series(equity).cummax() - 1.0
        dd_mask = dd_series < -0.05
        avg_exp_dd = np.nan  # requires full stored weights; proxy via abs return scale
        if dd_mask.any():
            avg_exp_dd = float(np.mean(np.abs(rr[dd_mask].to_numpy())))

        router_rows.append(
            {
                "split": split_id,
                "window": "oos",
                "start": str(df.index[start].date()),
                "end": str(df.index[test_end - 1].date()),
                "sharpe": _rolling_sharpe(strategy_ret),
                "calmar": _calmar(strategy_ret),
                "cvar_95": _cvar_95(strategy_ret),
                "bh_sharpe": _rolling_sharpe(bh_ret),
                "bh_calmar": _calmar(bh_ret),
                "bh_cvar_95": _cvar_95(bh_ret),
                "excess_sharpe": _rolling_sharpe(strategy_ret) - _rolling_sharpe(bh_ret),
                "regime_sharpe_high_vol": perf_high,
                "regime_sharpe_low_vol": perf_low,
                "switch_count": switched,
                "switch_rate": switched / max(len(selected), 1),
                "drawdown": _max_drawdown(equity),
                "bh_drawdown": _max_drawdown(bh_equity),
                "dd_behavior_proxy": avg_exp_dd,
                "best_train_pair": best_pair,
                "router_type": str(cfg.get("router", {}).get("type", "base")),
            }
        )

        split_elapsed = time.time() - split_t0
        total_elapsed = time.time() - t0
        print(
            f"[split {split_id:03d}] "
            f"{df.index[start].date()} -> {df.index[test_end - 1].date()} | "
            f"best_pair={best_pair} | "
            f"split_time={split_elapsed:.1f}s | total_time={total_elapsed:.1f}s"
        )

        start += step_days
        split_id += 1

    oos = np.array(oos_returns_all, dtype=float)
    oos_bh = np.array(oos_bh_returns_all, dtype=float)
    summary = pd.DataFrame(
        [
            {
                "overall_sharpe": _rolling_sharpe(oos),
                "overall_calmar": _calmar(oos),
                "overall_cvar_95": _cvar_95(oos),
                "bh_overall_sharpe": _rolling_sharpe(oos_bh),
                "bh_overall_calmar": _calmar(oos_bh),
                "bh_overall_cvar_95": _cvar_95(oos_bh),
                "excess_sharpe": _rolling_sharpe(oos) - _rolling_sharpe(oos_bh),
                "n_oos_obs": int(len(oos)),
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    est_df = pd.DataFrame(est_rows)
    ctrl_df = pd.DataFrame(ctrl_rows)
    router_df = pd.DataFrame(router_rows)

    est_df.to_csv(output_dir / "estimator_diagnostics.csv", index=False)
    ctrl_df.to_csv(output_dir / "controller_diagnostics.csv", index=False)
    router_df.to_csv(output_dir / "router_oos_diagnostics.csv", index=False)
    pd.DataFrame(oos_ts_rows).to_csv(output_dir / "router_vs_bh_timeseries.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    return {
        "estimator": est_df,
        "controller": ctrl_df,
        "router": router_df,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate rule-constrained router protocol")
    parser.add_argument("--strategy", "-s", default="configs/strategies/router_master.yaml")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Env.path("evaluation") / "router_protocol"),
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    tables = evaluate_router_protocol(
        config_path=args.strategy,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        max_pairs=args.max_pairs,
        output_dir=out,
    )

    print("=" * 60)
    print("Router protocol evaluation finished")
    print(f"Output dir: {out}")
    print(tables["summary"].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
