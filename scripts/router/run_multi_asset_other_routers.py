"""Run non-AI routers on the multi-asset pair universe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.router.bandit_router import ContextualBanditRouter
from src.router.moe_router import MixtureOfExpertsRouter
from src.router.router import Router
from src.router.rule_constraint_router import RuleConstraintRouter
from src.router.strategy_pair import StrategyPair
from src.router.ai_portfolio_regime_router import load_pair_results


class Component:
    """Small named component used by StrategyPair."""

    def __init__(self, name: str):
        self.name = name


def _component(name: str) -> Any:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    cls = type(safe[:80] or "Component", (Component,), {})
    return cls(name)


def _split_pair_name(name: str) -> tuple[str, str]:
    if "__" not in name:
        return name, ""
    return tuple(name.split("__", 1))  # type: ignore[return-value]


def _cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return 0.0
    q = float(r.quantile(alpha))
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) else q


def _metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, Any]:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return {
            "obs": 0,
            "rolling_sharpe": 0.0,
            "drawdown": 0.0,
            "annualized_return": 0.0,
            "realized_vol": 0.0,
            "vol_tracking_error": -0.10,
            "cvar_5": 0.0,
            "turnover": 0.0,
        }
    eq = np.exp(r.cumsum())
    ann_return = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "obs": int(len(r)),
        "rolling_sharpe": ann_return / ann_vol if ann_vol > 0 else 0.0,
        "drawdown": abs(min(dd, 0.0)),
        "annualized_return": ann_return,
        "realized_vol": ann_vol,
        "vol_tracking_error": ann_vol - 0.10,
        "cvar_5": abs(_cvar(r)),
        "turnover": float(pd.Series(turnover).dropna().mean()) if turnover is not None and len(pd.Series(turnover).dropna()) else 0.0,
    }


def _equity_metrics(returns: pd.Series) -> dict[str, float]:
    r = pd.Series(returns).fillna(0.0).astype(float)
    eq = np.exp(r.cumsum())
    ann_return = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    return {
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else 0.0,
        "max_drawdown": abs(float((eq / eq.cummax() - 1.0).min())) if len(eq) else 0.0,
        "cvar_95": abs(_cvar(r)),
        "n": float(len(r)),
    }


def _load_regimes(path: str | Path, index: pd.DatetimeIndex) -> pd.Series:
    p = Path(path)
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    col = "portfolio_regime" if "portfolio_regime" in df.columns else "regime"
    return df[col].astype(str).reindex(index).ffill().fillna("balanced")


def _market_features(date: pd.Timestamp, returns_hist: pd.Series, regime: str) -> dict[str, Any]:
    vol_map = {"risk_on": "low", "balanced": "middle", "defensive": "high"}
    r = returns_hist.dropna().astype(float)
    return {
        "date": str(date.date()),
        "portfolio_regime": regime,
        "vol_regime": vol_map.get(regime, "middle"),
        "rolling_vol": float(r.tail(63).std(ddof=1) * np.sqrt(252)) if len(r.tail(63)) > 1 else 0.0,
        "rolling_mean": float(r.tail(63).mean() * 252) if len(r.tail(63)) else 0.0,
        "rolling_skew": float(r.tail(63).skew()) if len(r.tail(63)) > 2 else 0.0,
        "window_obs": int(len(r.tail(63))),
    }


def _build_pairs(frames: dict[str, pd.DataFrame]) -> list[StrategyPair]:
    pairs: list[StrategyPair] = []
    for name in sorted(frames):
        estimator, controller = _split_pair_name(name)
        pairs.append(
            StrategyPair(
                name=name,
                estimator=_component(estimator),
                controller=_component(controller),
                metadata={
                    "estimator_path": estimator,
                    "controller_path": controller,
                },
            )
        )
    return pairs


def _router_params(router_name: str) -> dict[str, Any]:
    common = {
        "sticky_period": 10,
        "min_performance_obs": 20,
        "excluded_pairs_containing": ["__mean_variance", "buy_and_hold"],
        "lambda_switch": 0.05,
        "lambda_drawdown": 1.0,
        "use_heuristic_regime_bias": True,
    }
    if router_name == "base":
        return {**common, "perf_weight": 1.0, "regime_bias_weight": 0.2}
    if router_name == "rule_constraint":
        return {
            **common,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 0.15,
            "eta": 0.3,
            "kappa": 0.0,
            "max_turnover": 1.0,
            "max_vol_tracking_error": 0.20,
            "max_invalid_rate": 1.0,
            "max_exception_rate": 1.0,
        }
    if router_name == "contextual_bandit":
        return {
            **common,
            "alpha_ucb": 0.35,
            "lambda_risk": 0.8,
            "lambda_cost": 0.15,
            "lambda_vol_tracking": 0.3,
            "lambda_diag": 0.0,
            "min_arm_pulls": 0,
        }
    if router_name == "moe_conservative":
        return {
            **common,
            "temperature": 0.35,
            "momentum": 0.85,
            "switch_margin": 0.03,
            "lambda_risk": 1.0,
            "lambda_cost": 0.15,
            "lambda_vol_tracking": 0.3,
            "lambda_diag": 0.0,
        }
    if router_name == "moe_fast":
        return {
            **common,
            "sticky_period": 5,
            "lambda_switch": 0.01,
            "temperature": 0.25,
            "momentum": 0.55,
            "switch_margin": 0.005,
            "lambda_risk": 0.9,
            "lambda_cost": 0.10,
            "lambda_vol_tracking": 0.25,
            "lambda_diag": 0.0,
        }
    if router_name == "moe_aggressive":
        return {
            **common,
            "sticky_period": 3,
            "lambda_switch": 0.0,
            "temperature": 0.18,
            "momentum": 0.35,
            "switch_margin": 0.0,
            "lambda_risk": 0.8,
            "lambda_cost": 0.08,
            "lambda_vol_tracking": 0.2,
            "lambda_diag": 0.0,
        }
    if router_name in {"moe", "moe_return_tilt"}:
        return {
            **common,
            "sticky_period": 5,
            "lambda_switch": 0.0,
            "temperature": 0.22,
            "momentum": 0.50,
            "switch_margin": 0.0,
            "lambda_risk": 0.6,
            "lambda_cost": 0.05,
            "lambda_vol_tracking": 0.15,
            "lambda_diag": 0.0,
        }
    if router_name == "moe_reactive":
        return {
            **common,
            "sticky_period": 5,
            "lambda_switch": 0.0,
            "temperature": 0.28,
            "momentum": 0.65,
            "switch_margin": 0.0,
            "lambda_risk": 1.0,
            "lambda_cost": 0.10,
            "lambda_vol_tracking": 0.25,
            "lambda_diag": 0.0,
        }
    raise ValueError(router_name)


def _router_class(router_name: str):
    return {
        "base": Router,
        "rule_constraint": RuleConstraintRouter,
        "contextual_bandit": ContextualBanditRouter,
        "moe": MixtureOfExpertsRouter,
        "moe_conservative": MixtureOfExpertsRouter,
        "moe_aggressive": MixtureOfExpertsRouter,
        "moe_fast": MixtureOfExpertsRouter,
        "moe_return_tilt": MixtureOfExpertsRouter,
        "moe_reactive": MixtureOfExpertsRouter,
    }[router_name]


def run_router(
    router_name: str,
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    route_index: pd.DatetimeIndex,
    regimes: pd.Series,
    equal_weight_returns: pd.Series,
    out_dir: Path,
    initial_pair: str | None = None,
    initial_hold_days: int = 0,
) -> pd.DataFrame:
    pairs = _build_pairs(frames)
    router = _router_class(router_name)(pairs, _router_params(router_name))
    records: list[dict[str, Any]] = []
    router_returns: list[float] = []
    bh_returns: list[float] = []

    for date in route_index:
        loc = int(common_index.get_loc(date))
        hist_end = loc - 1
        perf: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        for name, frame in frames.items():
            hist = frame.iloc[max(0, hist_end - 63 + 1) : hist_end + 1] if hist_end >= 0 else frame.iloc[:0]
            metrics = _metrics(
                hist["returns_with_rf"],
                hist["turnover"] if "turnover" in hist else None,
            )
            perf[name] = metrics
            diagnostics[name] = {
                "obs": metrics["obs"],
                "turnover": metrics["turnover"],
                "vol_tracking_error": abs(metrics["vol_tracking_error"]),
                "estimator_loss": 0.0,
                "invalid_rate": 0.0,
                "exception_rate": 0.0,
            }
        ew_hist = equal_weight_returns.loc[:date].iloc[:-1]
        market = _market_features(date, ew_hist, str(regimes.loc[date]))
        route_step = len(records)
        initial_hold_active = bool(
            initial_pair in frames
            and initial_hold_days > 0
            and route_step < initial_hold_days
        )
        if initial_hold_active:
            selected_name = str(initial_pair)
            switched = False
        else:
            selected = router.select(market, diagnostics, perf, timestamp=date)
            selected_name = selected.name
            switched = bool(router.decisions[-1].get("switched", False))
        raw_ret = float(frames[selected_name].loc[date, "returns_with_rf"])
        router_returns.append(raw_ret)
        bh_ret = float(equal_weight_returns.loc[date])
        bh_returns.append(bh_ret)
        records.append(
            {
                "date": date,
                "router_signal_as_of_date": str(common_index[hist_end])[:10] if hist_end >= 0 else None,
                "strategy_return": raw_ret,
                "strategy_equity": float(np.exp(np.sum(router_returns))),
                "buy_hold_return": bh_ret,
                "buy_hold_equity": float(np.exp(np.sum(bh_returns))),
                "selected_pair": selected_name,
                "regime": str(regimes.loc[date]),
                "selected_turnover": float(frames[selected_name].get("turnover", pd.Series(0.0, index=frames[selected_name].index)).loc[date]),
                "selected_realized_vol": perf[selected_name]["realized_vol"],
                "selected_sharpe": perf[selected_name]["rolling_sharpe"],
                "selected_drawdown": perf[selected_name]["drawdown"],
                "switched": switched,
                "initial_hold_active": initial_hold_active,
            }
        )

    out = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "router_vs_bh_timeseries.csv", index=False)
    out.to_parquet(out_dir / "router_vs_bh_timeseries.parquet", index=False)

    router_m = _equity_metrics(out["strategy_return"])
    bh_m = _equity_metrics(out["buy_hold_return"])
    summary = pd.DataFrame(
        [
            {"strategy": router_name, **router_m, "switches": int(out["switched"].sum()), "pairs_used": int(out["selected_pair"].nunique())},
            {"strategy": "equal_weight_buy_hold", **bh_m, "switches": 0, "pairs_used": 1},
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    router_diag = pd.DataFrame(
        [
            {
                "router": router_name,
                "rows": len(out),
                "switches": int(out["switched"].sum()),
                "pairs_used": int(out["selected_pair"].nunique()),
                "avg_selected_turnover": float(out["selected_turnover"].mean()),
                "avg_selected_realized_vol": float(out["selected_realized_vol"].mean()),
            }
        ]
    )
    router_diag.to_csv(out_dir / "router_oos_diagnostics.csv", index=False)

    pair_diag = (
        out.groupby("selected_pair")
        .agg(
            days=("selected_pair", "size"),
            avg_turnover=("selected_turnover", "mean"),
            avg_realized_vol=("selected_realized_vol", "mean"),
            avg_sharpe=("selected_sharpe", "mean"),
            max_drawdown_proxy=("selected_drawdown", "max"),
        )
        .reset_index()
    )
    pair_diag["estimator"] = pair_diag["selected_pair"].str.split("__").str[0]
    pair_diag["controller"] = pair_diag["selected_pair"].str.split("__").str[1]
    pair_diag.groupby("estimator").agg(days=("days", "sum"), pairs=("selected_pair", "nunique")).reset_index().to_csv(
        out_dir / "estimator_diagnostics.csv",
        index=False,
    )
    pair_diag.groupby("controller").agg(days=("days", "sum"), pairs=("selected_pair", "nunique")).reset_index().to_csv(
        out_dir / "controller_diagnostics.csv",
        index=False,
    )
    pair_diag.to_csv(out_dir / "pair_usage_diagnostics.csv", index=False)
    return summary.iloc[[0]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/multi_asset_tuned_pairs_vol10/manifest.csv")
    parser.add_argument("--out-dir", default="results/evaluation/multi_asset_other_routers_cooldown60_setup")
    parser.add_argument("--regime-path", default="results/multi_asset_tuned_pairs_vol10/ai_volatility_regime_series_macro_inputs_start20230210_interval10.csv")
    parser.add_argument("--returns-path", default="data/processed/emm_daily_log_returns_yahoo_20220210_20260210.parquet")
    parser.add_argument("--metric-start-date", default="2023-02-10")
    parser.add_argument("--start-date", default="2024-02-09")
    parser.add_argument("--routers", default="base,rule_constraint,contextual_bandit,moe")
    parser.add_argument("--initial-pair", default=None)
    parser.add_argument("--initial-hold-days", type=int, default=0)
    args = parser.parse_args()

    frames = load_pair_results(args.manifest)
    common_index = pd.DatetimeIndex(sorted(set.intersection(*[set(df.index) for df in frames.values()])))
    common_index = common_index[common_index >= pd.Timestamp(args.metric_start_date)]
    route_index = common_index[common_index >= pd.Timestamp(args.start_date)]
    frames = {name: df.loc[common_index].copy() for name, df in frames.items()}

    asset_returns = pd.read_parquet(args.returns_path).sort_index()
    asset_returns.index = pd.to_datetime(asset_returns.index)
    equal_weight_returns = asset_returns.mean(axis=1).reindex(common_index).fillna(0.0)
    regimes = _load_regimes(args.regime_path, common_index)

    out_base = Path(args.out_dir)
    summaries = []
    for router_name in [r.strip() for r in args.routers.split(",") if r.strip()]:
        print(f"running {router_name}")
        summaries.append(
            run_router(
                router_name,
                frames,
                common_index,
                route_index,
                regimes,
                equal_weight_returns,
                out_base / router_name,
                initial_pair=args.initial_pair,
                initial_hold_days=args.initial_hold_days,
            )
        )
    comparison = pd.concat(summaries, ignore_index=True)
    comparison.to_csv(out_base / "comparison_summary.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
