"""Run non-AI multi-asset router and benchmark suite under the current universe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.run_multi_asset_other_routers import (  # noqa: E402
    _build_pairs,
    _load_regimes,
    _market_features,
    _metrics as router_window_metrics,
    _router_class,
    _router_params,
)
from src.router.ai_portfolio_regime_router import load_pair_results  # noqa: E402


EXCLUDE_TOKENS = [
    "minimum_variance",
    "min_variance",
    "mean_variance",
    "buy_and_hold",
    "equal_weight",
]


def cvar_95(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return 0.0
    q = float(r.quantile(0.05))
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) else q


def performance_table_row(
    label: str,
    type_name: str,
    returns: pd.Series,
    active_pairs: pd.Series | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        raise ValueError(f"No returns for {label}")
    equity = np.exp(r.cumsum())
    drawdown = equity / equity.cummax() - 1.0
    ann_return = float(r.mean() * 252.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252.0)) if len(r) > 1 else 0.0
    downside = r[r < 0.0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(252.0)) if len(downside) > 1 else 0.0
    max_dd = float(drawdown.min())
    switches = 0
    unique_pair_use = 1
    if active_pairs is not None:
        pairs = pd.Series(active_pairs).astype(str)
        switches = int((pairs != pairs.shift(1)).sum() - 1)
        unique_pair_use = int(pairs.nunique())
    row = {
        "type": type_name,
        "method": label,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else 0.0,
        "max_dd": max_dd,
        "cvar95": cvar_95(r),
        "switch_count": switches,
        "unique_pair_use": unique_pair_use,
        "sortino": ann_return / downside_vol if downside_vol else 0.0,
        "calmar": ann_return / abs(max_dd) if max_dd else 0.0,
        "total_return": float(equity.iloc[-1] - 1.0),
        "rows": int(len(r)),
        "start": str(r.index.min())[:10],
        "end": str(r.index.max())[:10],
    }
    if extra:
        row.update(extra)
    return row


def save_timeseries(
    out_dir: Path,
    slug: str,
    returns: pd.Series,
    active_pair: pd.Series | str,
    regime: pd.Series | None = None,
) -> None:
    r = pd.Series(returns).dropna().astype(float)
    if isinstance(active_pair, str):
        pair_series = pd.Series(active_pair, index=r.index)
    else:
        pair_series = pd.Series(active_pair).reindex(r.index).astype(str)
    out = pd.DataFrame(
        {
            "returns_with_rf": r,
            "equity_curve_with_rf": 1000.0 * np.exp(r.cumsum()),
            "active_pair": pair_series,
        },
        index=r.index,
    )
    if regime is not None:
        out["portfolio_regime"] = pd.Series(regime).reindex(r.index).astype(str)
    out.index.name = "date"
    out.to_parquet(out_dir / f"{slug}.parquet")
    out.to_csv(out_dir / f"{slug}.csv")


def load_current_pair_universe(manifest: str, metric_start: str) -> dict[str, pd.DataFrame]:
    frames = load_pair_results(manifest)
    frames = {
        name: frame
        for name, frame in frames.items()
        if not any(token in name for token in EXCLUDE_TOKENS)
    }
    common_index = pd.DatetimeIndex(sorted(set.intersection(*[set(df.index) for df in frames.values()])))
    common_index = common_index[common_index >= pd.Timestamp(metric_start)]
    return {name: df.loc[common_index].copy() for name, df in frames.items()}


def train_score(frame: pd.DataFrame) -> float:
    r = frame["returns_with_rf"].dropna().astype(float)
    if len(r) < 60:
        return -np.inf
    equity = np.exp(r.cumsum())
    dd = abs(float((equity / equity.cummax() - 1.0).min()))
    ann_return = float(r.mean() * 252.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252.0))
    sharpe = ann_return / ann_vol if ann_vol else 0.0
    return float(sharpe + 0.25 * ann_return - 0.75 * dd - 2.0 * abs(cvar_95(r)))


def best_fixed_pair(frames: dict[str, pd.DataFrame], route_index: pd.DatetimeIndex) -> tuple[str, pd.Series]:
    rows = []
    for name, frame in frames.items():
        row = performance_table_row(name, "fixed_candidate", frame.loc[route_index, "returns_with_rf"])
        rows.append(row)
    ranked = pd.DataFrame(rows).sort_values(["sharpe", "max_dd"], ascending=[False, False])
    pair = str(ranked.iloc[0]["method"])
    return pair, frames[pair].loc[route_index, "returns_with_rf"]


def requested_fixed_pair(
    frames: dict[str, pd.DataFrame],
    pair: str,
    route_index: pd.DatetimeIndex,
) -> tuple[str, pd.Series]:
    if pair not in frames:
        raise ValueError(f"Requested fixed pair is not in current universe: {pair}")
    return pair, frames[pair].loc[route_index, "returns_with_rf"]


def train_champion_pair(
    frames: dict[str, pd.DataFrame],
    train_index: pd.DatetimeIndex,
) -> tuple[str, pd.Series]:
    scores = {
        name: train_score(frame.loc[train_index])
        for name, frame in frames.items()
    }
    pair = max(scores, key=scores.get)
    return pair, frames[pair]["returns_with_rf"]


def equal_weight_asset_returns(path: str, index: pd.DatetimeIndex) -> pd.Series:
    asset_returns = pd.read_parquet(path).sort_index()
    asset_returns.index = pd.to_datetime(asset_returns.index)
    return asset_returns.mean(axis=1).reindex(index).fillna(0.0)


def vol_scaled_returns(
    base_returns: pd.Series,
    target_vol: float,
    mode: str,
    window: int = 22,
    span: int = 22,
    cap: float = 2.0,
    smooth: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    r = pd.Series(base_returns).fillna(0.0).astype(float)
    if mode == "rv":
        vol = r.rolling(window).std(ddof=1) * np.sqrt(252.0)
    elif mode == "ewma":
        vol = r.ewm(span=span, adjust=False).std(bias=False) * np.sqrt(252.0)
    else:
        raise ValueError(mode)
    raw_lev = (target_vol / vol.shift(1).replace(0.0, np.nan)).clip(lower=0.0, upper=cap).fillna(1.0)
    if smooth is not None:
        vals = []
        prev = float(raw_lev.iloc[0])
        alpha = float(smooth)
        for value in raw_lev:
            prev = alpha * prev + (1.0 - alpha) * float(value)
            vals.append(min(prev, cap))
        lev = pd.Series(vals, index=r.index)
    else:
        lev = raw_lev
    return r * lev, lev


def smoothed_capped_ewma_pair_returns(
    ewma_pair: pd.DataFrame,
    cap: float = 1.25,
    smooth: float = 0.80,
) -> tuple[pd.Series, pd.Series]:
    """Use the EWMA covariance equal-weight pair, but smooth/cap its gross exposure."""
    frame = ewma_pair.copy()
    gross = frame["gross_exposure"].astype(float).clip(lower=0.0, upper=float(cap))
    smoothed = []
    prev = float(gross.iloc[0])
    for value in gross:
        prev = float(smooth) * prev + (1.0 - float(smooth)) * float(value)
        smoothed.append(min(prev, float(cap)))
    leverage = pd.Series(smoothed, index=frame.index)
    raw_risky = frame["risky_leg_return"].astype(float) / frame["gross_exposure"].replace(0.0, np.nan).astype(float)
    raw_risky = raw_risky.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rf = frame.get("rf_daily_return", pd.Series(0.0, index=frame.index)).astype(float)
    returns = leverage * raw_risky + (1.0 - leverage).clip(lower=0.0) * rf
    return returns, leverage


def run_dynamic_pair_router(
    router_name: str,
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    route_index: pd.DatetimeIndex,
    regimes: pd.Series,
    equal_weight_returns: pd.Series,
    out_dir: Path,
) -> tuple[pd.Series, pd.Series]:
    pairs = _build_pairs(frames)
    params = _router_params(router_name)
    params["excluded_pairs_containing"] = []
    router = _router_class(router_name)(pairs, params)
    returns = []
    selected_pairs = []
    for date in route_index:
        loc = int(common_index.get_loc(date))
        hist_end = loc - 1
        perf: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        for name, frame in frames.items():
            hist = frame.iloc[max(0, hist_end - 63 + 1) : hist_end + 1] if hist_end >= 0 else frame.iloc[:0]
            metrics = router_window_metrics(
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
        selected = router.select(market, diagnostics, perf, timestamp=date)
        selected_pairs.append(selected.name)
        returns.append(float(frames[selected.name].loc[date, "returns_with_rf"]))
    return pd.Series(returns, index=route_index), pd.Series(selected_pairs, index=route_index)


def run_recent_ensemble(
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    route_index: pd.DatetimeIndex,
    train_index: pd.DatetimeIndex,
    top_n: int = 8,
    review_interval: int = 10,
    lookback: int = 63,
) -> tuple[pd.Series, pd.Series]:
    train_ranked = sorted(
        ((train_score(frame.loc[train_index]), name) for name, frame in frames.items()),
        reverse=True,
    )
    pool = [name for _, name in train_ranked[: max(top_n * 2, top_n)]]
    active = pool[:top_n]
    records = []
    active_labels = []
    for step, date in enumerate(route_index):
        loc = int(common_index.get_loc(date))
        if step == 0 or step % review_interval == 0:
            rows = []
            for name in pool:
                hist = frames[name].iloc[max(0, loc - lookback) : loc]
                rows.append((train_score(hist), name))
            active = [name for _, name in sorted(rows, reverse=True)[:top_n]]
        day_returns = [float(frames[name].loc[date, "returns_with_rf"]) for name in active]
        records.append(float(np.mean(day_returns)))
        active_labels.append("|".join(active))
    return pd.Series(records, index=route_index), pd.Series(active_labels, index=route_index)


def run_regime_fixed_mapping(
    frames: dict[str, pd.DataFrame],
    train_index: pd.DatetimeIndex,
    route_index: pd.DatetimeIndex,
    regimes: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    mapping: dict[str, str] = {}
    for regime in sorted(set(regimes.dropna().astype(str))):
        idx = train_index[regimes.reindex(train_index).astype(str) == regime]
        if len(idx) < 40:
            idx = train_index
        scores = {name: train_score(frame.loc[idx]) for name, frame in frames.items()}
        mapping[regime] = max(scores, key=scores.get)
    selected = regimes.reindex(route_index).astype(str).map(mapping).fillna(next(iter(mapping.values())))
    returns = pd.Series(
        [float(frames[str(pair)].loc[date, "returns_with_rf"]) for date, pair in selected.items()],
        index=route_index,
    )
    return returns, selected, mapping


def write_switch_log(out_dir: Path, slug: str, active_pairs: pd.Series) -> None:
    pairs = pd.Series(active_pairs).astype(str)
    changes = pairs[pairs != pairs.shift(1)].rename("active_pair").reset_index()
    changes.columns = ["date", "active_pair"]
    changes.to_csv(out_dir / f"{slug}_switch_log.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/multi_asset_tuned_pairs_vol10/manifest.csv")
    parser.add_argument("--returns-path", default="data/processed/emm_daily_log_returns_yahoo_20220210_20260210.parquet")
    parser.add_argument("--regime-path", default="results/multi_asset_tuned_pairs_vol10/ai_volatility_regime_series_macro_inputs_start20230210_interval10_noleak.csv")
    parser.add_argument("--out-dir", default="results/multi_asset_tuned_pairs_vol10/non_ai_router_benchmark_suite_current_universe")
    parser.add_argument("--metric-start-date", default="2023-02-10")
    parser.add_argument("--train-end", default="2024-02-08")
    parser.add_argument("--start-date", default="2024-02-09")
    parser.add_argument("--end-date", default="2026-02-10")
    parser.add_argument("--fixed-pair", default="expanding_covariance__hysteresis_portfolio")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_current_pair_universe(args.manifest, args.metric_start_date)
    common_index = pd.DatetimeIndex(sorted(set.intersection(*[set(df.index) for df in frames.values()])))
    route_index = common_index[
        (common_index >= pd.Timestamp(args.start_date))
        & (common_index <= pd.Timestamp(args.end_date))
    ]
    train_index = common_index[
        (common_index >= pd.Timestamp(args.metric_start_date))
        & (common_index <= pd.Timestamp(args.train_end))
    ]
    regimes = _load_regimes(args.regime_path, common_index)
    equal_weight_returns = equal_weight_asset_returns(args.returns_path, common_index)
    rows: list[dict[str, Any]] = []

    metadata = {
        "pair_universe_size": len(frames),
        "excluded_tokens": EXCLUDE_TOKENS,
        "metric_start_date": args.metric_start_date,
        "train_end": args.train_end,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "regime_path": args.regime_path,
        "returns_path": args.returns_path,
        "definitions": {
            "buy_and_hold": "sample_covariance__buy_and_hold, equal initial asset weights after first-day vol scaling, then held constant.",
            "rv22_naive_scaling": "Equal-weight asset basket scaled by target_vol / past 22-day realized vol, using t-1 vol and max leverage 2.0.",
            "ewma_vol_targeting": "Actual ewma_covariance__equal_weight pair: EWMA covariance estimate plus equal-weight naive volatility scaling.",
            "smoothed_capped_vol_targeting": "Uses ewma_covariance__equal_weight raw risky return, but applies 0.80-smoothed EWMA gross exposure capped at 1.25.",
            "ensemble_router": "Every 10 days, average the current top 8 pairs from the train-selected top pool by trailing 63-day score.",
            "regime_aware_fixed_mapping": "Pick one train-window champion pair per precomputed volatility regime, then map OOS dates by that regime.",
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    buy_hold_path = Path("results/multi_asset_tuned_pairs_vol10/sample_covariance__buy_and_hold.parquet")
    buy_hold = pd.read_parquet(buy_hold_path).loc[route_index, "returns_with_rf"]
    rows.append(performance_table_row("Vol-scaled equal-weight buy-and-hold", "baseline", buy_hold, extra={"source": str(buy_hold_path)}))
    save_timeseries(out_dir, "buy_and_hold_equal_initial_weight", buy_hold, "sample_covariance__buy_and_hold", regimes.reindex(route_index))

    rv_returns, rv_lev = vol_scaled_returns(equal_weight_returns, 0.10, mode="rv", window=22, cap=2.0)
    rows.append(performance_table_row("RV22 + NaiveScaling equivalent", "vol_target_benchmark", rv_returns.loc[route_index], extra={"avg_leverage": float(rv_lev.loc[route_index].mean())}))
    save_timeseries(out_dir, "rv22_naive_scaling_equivalent", rv_returns.loc[route_index], "equal_weight_assets__rv22_naive_scaling", regimes.reindex(route_index))

    ewma_pair_path = Path("results/multi_asset_tuned_pairs_vol10/ewma_covariance__equal_weight.parquet")
    ewma_pair = pd.read_parquet(ewma_pair_path).loc[common_index]
    ewma_returns = ewma_pair["returns_with_rf"].astype(float)
    rows.append(
        performance_table_row(
            "EWMA covariance + NaiveScaling",
            "vol_target_benchmark",
            ewma_returns.loc[route_index],
            extra={
                "avg_leverage": float(ewma_pair.loc[route_index, "gross_exposure"].mean()),
                "source": str(ewma_pair_path),
            },
        )
    )
    save_timeseries(out_dir, "ewma_covariance_naive_scaling", ewma_returns.loc[route_index], "ewma_covariance__equal_weight", regimes.reindex(route_index))

    smooth_returns, smooth_lev = smoothed_capped_ewma_pair_returns(ewma_pair, cap=1.25, smooth=0.80)
    rows.append(
        performance_table_row(
            "Smoothed / capped EWMA covariance vol targeting",
            "vol_target_benchmark",
            smooth_returns.loc[route_index],
            extra={
                "avg_leverage": float(smooth_lev.loc[route_index].mean()),
                "source": str(ewma_pair_path),
            },
        )
    )
    save_timeseries(out_dir, "smoothed_capped_ewma_covariance_vol_targeting", smooth_returns.loc[route_index], "ewma_covariance__smoothed_capped_equal_weight", regimes.reindex(route_index))

    fixed_pair, fixed_returns = requested_fixed_pair(frames, args.fixed_pair, route_index)
    rows.append(
        performance_table_row(
            f"Fixed pair: {fixed_pair}",
            "fixed_pair",
            fixed_returns,
            fixed_pair,
            {"selection_basis": "user_specified_fixed_benchmark"},
        )
    )
    save_timeseries(out_dir, "user_specified_fixed_pair", fixed_returns, fixed_pair, regimes.reindex(route_index))

    best_pair, best_returns = best_fixed_pair(frames, route_index)
    pd.DataFrame(
        [
            performance_table_row(
                f"Best fixed pair ex-post: {best_pair}",
                "fixed_pair_reference",
                best_returns,
                best_pair,
                {"selection_basis": "ex_post_reference"},
            )
        ]
    ).to_csv(out_dir / "best_fixed_pair_ex_post_reference.csv", index=False)
    save_timeseries(out_dir, "best_fixed_pair_ex_post_reference", best_returns, best_pair, regimes.reindex(route_index))

    regime_returns, regime_pairs, mapping = run_regime_fixed_mapping(frames, train_index, route_index, regimes)
    rows.append(performance_table_row("Regime-aware fixed mapping", "regime_aware", regime_returns, regime_pairs, {"mapping": json.dumps(mapping, sort_keys=True)}))
    save_timeseries(out_dir, "regime_aware_fixed_mapping", regime_returns, regime_pairs, regimes.reindex(route_index))
    write_switch_log(out_dir, "regime_aware_fixed_mapping", regime_pairs)

    ensemble_returns, ensemble_pairs = run_recent_ensemble(frames, common_index, route_index, train_index)
    rows.append(performance_table_row("Ensemble router: rolling top-8 average", "ensemble", ensemble_returns, ensemble_pairs))
    save_timeseries(out_dir, "ensemble_router_rolling_top8_average", ensemble_returns, ensemble_pairs, regimes.reindex(route_index))
    write_switch_log(out_dir, "ensemble_router_rolling_top8_average", ensemble_pairs)

    for router_name, label in [
        ("base", "Base deterministic router"),
        ("rule_constraint", "Rule-based constrained router"),
        ("contextual_bandit", "Contextual bandit over pairs"),
        ("moe", "Mixture-of-experts router"),
    ]:
        returns, selected = run_dynamic_pair_router(
            router_name,
            frames,
            common_index,
            route_index,
            regimes,
            equal_weight_returns,
            out_dir,
        )
        rows.append(performance_table_row(label, "router", returns, selected))
        save_timeseries(out_dir, router_name, returns, selected, regimes.reindex(route_index))
        write_switch_log(out_dir, router_name, selected)

    table = pd.DataFrame(rows)
    ordered = [
        "type",
        "method",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_dd",
        "cvar95",
        "switch_count",
        "unique_pair_use",
        "sortino",
        "calmar",
        "total_return",
        "rows",
        "start",
        "end",
    ]
    extras = [col for col in table.columns if col not in ordered]
    table = table[ordered + extras]
    table.to_csv(out_dir / "non_ai_router_benchmark_table.csv", index=False)
    table.to_parquet(out_dir / "non_ai_router_benchmark_table.parquet", index=False)
    print(table.to_string(index=False))
    print(f"Wrote table to {out_dir / 'non_ai_router_benchmark_table.csv'}")


if __name__ == "__main__":
    main()
