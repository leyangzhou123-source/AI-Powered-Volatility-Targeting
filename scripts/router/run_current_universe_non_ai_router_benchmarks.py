"""Run non-AI router benchmarks on the current SPY pair universe."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.evaluate_router_protocol_precomputed import (  # noqa: E402
    _load_pair_results,
    _normalize_pair_result,
    evaluate_precomputed_router_protocol,
)
from scripts.router.run_ai_regime_router_all_pairs import _pair_cfg_is_excluded  # noqa: E402


CURRENT_ROUTER_EXCLUDED_ESTIMATORS = [
    "ar1",
    "ar2",
    "ewma",
    "hybrid_ewma",
    "hybrid_ewmaregime",
    "lasso",
    "lasso_volatility",
    "buy_and_hold",
    "buyandhold",
]

CURRENT_ROUTER_EXCLUDED_CONTROLLERS = [
    "constant",
    "constant_weight",
    "cva_restargeting",
    "cvar",
    "cvar_es",
    "cvares",
]


def cvar_95(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return 0.0
    q = float(r.quantile(0.05))
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) else q


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if r.empty:
        return 0.0
    equity = np.exp(r.cumsum())
    return float((equity / equity.cummax() - 1.0).min())


def metric_row(
    strategy: str,
    router_type: str,
    returns: pd.Series,
    selected_pair: pd.Series | str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        raise ValueError(f"No returns for {strategy}")
    ann_return = float(r.mean() * 252.0)
    ann_vol = float(r.std(ddof=1) * math.sqrt(252.0)) if len(r) > 1 else 0.0
    downside = r[r < 0.0]
    downside_vol = float(downside.std(ddof=1) * math.sqrt(252.0)) if len(downside) > 1 else 0.0
    dd = max_drawdown(r)

    if isinstance(selected_pair, str):
        pairs = pd.Series(selected_pair, index=r.index)
    else:
        pairs = pd.Series(selected_pair).reindex(r.index).astype(str)
    switch_count = int((pairs != pairs.shift(1)).sum() - 1) if len(pairs) else 0

    out = {
        "strategy": strategy,
        "router_type": router_type,
        "ann_vol": ann_vol,
        "ann_return": ann_return,
        "sharpe": ann_return / ann_vol if ann_vol else 0.0,
        "max_dd": dd,
        "cvar": cvar_95(r),
        "switch_count": switch_count,
        "unique_pair_use": int(pairs.nunique()) if len(pairs) else 0,
        "sortino": ann_return / downside_vol if downside_vol else 0.0,
        "calmar": ann_return / abs(dd) if dd else 0.0,
        "start": str(r.index.min())[:10],
        "end": str(r.index.max())[:10],
        "n_obs": int(len(r)),
    }
    if extra:
        out.update(extra)
    return out


def save_timeseries(out_dir: Path, slug: str, returns: pd.Series, selected_pair: pd.Series | str) -> None:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if isinstance(selected_pair, str):
        selected = pd.Series(selected_pair, index=r.index)
    else:
        selected = pd.Series(selected_pair).reindex(r.index).astype(str)
    out = pd.DataFrame(
        {
            "return": r,
            "equity": np.exp(r.cumsum()),
            "selected_pair": selected,
        },
        index=r.index,
    )
    out.index.name = "date"
    out.to_csv(out_dir / f"{slug}_timeseries.csv")
    out.to_parquet(out_dir / f"{slug}_timeseries.parquet")
    switches = out.loc[out["selected_pair"] != out["selected_pair"].shift(1), ["selected_pair"]]
    switches.to_csv(out_dir / f"{slug}_switch_log.csv")


def filtered_config(config_path: Path, router_type: str) -> tuple[dict[str, Any], int, int]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = dict((cfg.get("router", {}) or {}).get("params", {}) or {})
    params.update(
        {
            "excluded_estimators": CURRENT_ROUTER_EXCLUDED_ESTIMATORS,
            "excluded_controllers": CURRENT_ROUTER_EXCLUDED_CONTROLLERS,
            "fail_open": True,
            "retry_ai_after_failure": False,
            "sticky_period": 1,
            "min_performance_obs": 5,
            "use_perf_weight": True,
            "disable_regime_context": True,
            "disable_recent_rank_context": True,
            "disable_benchmark_context": True,
            "disable_deterministic_baseline_context": True,
        }
    )
    pairs_before = list((cfg.get("router", {}) or {}).get("pairs", []) or [])
    pairs_after = [pair for pair in pairs_before if not _pair_cfg_is_excluded(pair, params)]
    cfg.setdefault("router", {})["pairs"] = pairs_after
    cfg.setdefault("router", {})["type"] = router_type
    cfg.setdefault("router", {})["params"] = params
    return cfg, len(pairs_before), len(pairs_after)


def run_router(
    cfg: dict[str, Any],
    config_path: Path,
    pair_results_dir: Path,
    out_dir: Path,
    oos_start_date: str,
    oos_end_date: str,
) -> pd.DataFrame:
    tables = evaluate_precomputed_router_protocol(
        config_path=str(config_path),
        pair_results_dir=pair_results_dir,
        train_days=504,
        test_days=252,
        step_days=252,
        max_pairs=None,
        output_dir=out_dir,
        config=cfg,
        verbose=False,
        oos_start_date=oos_start_date,
        oos_end_date=oos_end_date,
        single_oos_window=True,
        freeze_oos_metrics=False,
    )
    return tables["router"]


def load_route_index(router_out_dir: Path) -> pd.DatetimeIndex:
    ts = pd.read_csv(router_out_dir / "router_vs_bh_timeseries.csv")
    return pd.DatetimeIndex(pd.to_datetime(ts["date"]))


def load_filtered_pair_results(
    cfg: dict[str, Any],
    pair_results_dir: Path,
    common_index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    frames = _load_pair_results(cfg, pair_results_dir)
    return {
        name: frame.loc[common_index].copy()
        for name, frame in frames.items()
        if set(common_index).issubset(set(frame.index))
    }


def filtered_common_index(cfg: dict[str, Any], pair_results_dir: Path) -> pd.DatetimeIndex:
    frames = _load_pair_results(cfg, pair_results_dir)
    common: pd.DatetimeIndex | None = None
    for frame in frames.values():
        idx = pd.DatetimeIndex(frame.index)
        common = idx if common is None else common.intersection(idx)
    if common is None:
        raise ValueError("No pair frames available.")
    return common.sort_values()


def best_fixed_pair(frames: dict[str, pd.DataFrame], route_index: pd.DatetimeIndex) -> tuple[str, pd.Series]:
    rows = []
    for name, frame in frames.items():
        returns = pd.to_numeric(frame.loc[route_index, "returns"], errors="coerce").fillna(0.0)
        rows.append(metric_row(name, "fixed_pair_candidate", returns, name))
    table = pd.DataFrame(rows).sort_values(
        ["sharpe", "calmar", "max_dd"],
        ascending=[False, False, False],
    )
    pair = str(table.iloc[0]["strategy"])
    return pair, pd.to_numeric(frames[pair].loc[route_index, "returns"], errors="coerce").fillna(0.0)


def train_score(frame: pd.DataFrame) -> float:
    returns = pd.to_numeric(frame["returns"], errors="coerce").fillna(0.0)
    if len(returns) < 60:
        return float("-inf")
    row = metric_row("candidate", "candidate", returns, "candidate")
    return float(
        row["sharpe"]
        + 0.25 * row["ann_return"]
        - 0.75 * abs(row["max_dd"])
        - 2.0 * abs(row["cvar"])
    )


def load_regime_series(path: Path, index: pd.DatetimeIndex) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    if "ai_vol_regime" in df.columns:
        col = "ai_vol_regime"
    elif "vol_regime" in df.columns:
        col = "vol_regime"
    elif "regime" in df.columns:
        col = "regime"
    else:
        raise ValueError(f"No regime column found in {path}")
    regimes = df[col].astype(str).str.lower().replace({"mid": "middle", "normal": "middle"})
    return regimes.reindex(index).ffill().bfill().fillna("middle")


def regime_aware_fixed_mapping(
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    route_index: pd.DatetimeIndex,
    regimes: pd.Series,
    train_days: int = 504,
) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    start_pos = int(common_index.get_loc(route_index[0]))
    train_index = common_index[max(0, start_pos - train_days) : start_pos]
    if len(train_index) == 0:
        raise ValueError("No train index available for regime-aware fixed mapping.")
    mapping: dict[str, str] = {}
    train_regimes = regimes.reindex(train_index)
    for regime in sorted(set(regimes.reindex(route_index).dropna().astype(str))):
        idx = train_index[train_regimes.astype(str) == regime]
        if len(idx) < 40:
            idx = train_index
        scores = {name: train_score(frame.loc[idx]) for name, frame in frames.items()}
        mapping[regime] = max(scores, key=scores.get)
    selected = regimes.reindex(route_index).astype(str).map(mapping)
    fallback = next(iter(mapping.values()))
    selected = selected.fillna(fallback)
    returns = pd.Series(
        [float(frames[str(pair)].loc[date, "returns"]) for date, pair in selected.items()],
        index=route_index,
    )
    return returns, selected, mapping


def vol_target_returns(
    asset_returns: pd.Series,
    rf_returns: pd.Series,
    target_vol: float,
    mode: str,
    window: int = 22,
    span: int = 22,
    cap: float = 1.5,
    smooth: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    asset = pd.Series(asset_returns).astype(float).fillna(0.0)
    rf = pd.Series(rf_returns).reindex(asset.index).astype(float).fillna(0.0)
    if mode == "rv":
        vol = asset.rolling(window).std(ddof=1) * math.sqrt(252.0)
    elif mode == "ewma":
        vol = asset.ewm(span=span, adjust=False).std(bias=False) * math.sqrt(252.0)
    else:
        raise ValueError(mode)
    raw_weight = (target_vol / vol.shift(1).replace(0.0, np.nan)).clip(lower=0.0, upper=cap)
    raw_weight = raw_weight.ffill().fillna(1.0)
    if smooth is not None:
        alpha = float(smooth)
        values: list[float] = []
        prev = float(raw_weight.iloc[0])
        for value in raw_weight:
            prev = alpha * prev + (1.0 - alpha) * float(value)
            values.append(min(max(prev, 0.0), cap))
        weight = pd.Series(values, index=asset.index)
    else:
        weight = raw_weight
    returns = weight * asset + (1.0 - weight) * rf
    return returns, weight


def router_table_row(name: str, router_type: str, out_dir: Path) -> dict[str, Any]:
    ts = pd.read_csv(out_dir / "router_vs_bh_timeseries.csv")
    ts["date"] = pd.to_datetime(ts["date"])
    returns = pd.Series(ts["router_return"].to_numpy(dtype=float), index=ts["date"])
    selected = pd.Series(ts["selected_pair"].astype(str).to_numpy(), index=ts["date"])
    return metric_row(name, router_type, returns, selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strategies/router_master.yaml")
    parser.add_argument("--pair-results-dir", default="results/all_estimator_controller_pairs")
    parser.add_argument("--output-dir", default="results/evaluation/current_universe_non_ai_router_benchmarks")
    parser.add_argument("--oos-start-date", default="2023-01-01")
    parser.add_argument("--oos-end-date", default="2026-02-10")
    parser.add_argument("--naive-scaling-path", default="results/realized_vol_22__naive_scaling_oos.parquet")
    parser.add_argument("--regime-path", default="results/evaluation/ai_regime_series/ai_regime_10d.csv")
    args = parser.parse_args()

    config_path = Path(args.config)
    pair_results_dir = Path(args.pair_results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "config": str(config_path),
        "pair_results_dir": str(pair_results_dir),
        "oos_start_date_requested": args.oos_start_date,
        "oos_end_date_requested": args.oos_end_date,
        "train_days": 504,
        "test_days": 252,
        "step_days": 252,
        "single_oos_window": True,
        "excluded_estimators": CURRENT_ROUTER_EXCLUDED_ESTIMATORS,
        "excluded_controllers": CURRENT_ROUTER_EXCLUDED_CONTROLLERS,
        "regime_path": args.regime_path,
    }

    router_runs = [
        ("ensemble_router_moe", "mixture_of_experts"),
        ("contextual_bandit_router", "contextual_bandit"),
    ]
    first_cfg: dict[str, Any] | None = None
    first_out: Path | None = None
    for slug, router_type in router_runs:
        cfg, pairs_before, pairs_after = filtered_config(config_path, router_type)
        if first_cfg is None:
            first_cfg = cfg
            metadata["pair_universe_before_filter"] = pairs_before
            metadata["pair_universe_after_filter"] = pairs_after
        run_dir = out_dir / slug
        print(f"running {slug} ({router_type}) on {pairs_after} pairs")
        run_router(
            cfg=cfg,
            config_path=config_path,
            pair_results_dir=pair_results_dir,
            out_dir=run_dir,
            oos_start_date=args.oos_start_date,
            oos_end_date=args.oos_end_date,
        )
        rows.append(router_table_row(slug, router_type, run_dir))
        if first_out is None:
            first_out = run_dir

    if first_cfg is None or first_out is None:
        raise RuntimeError("No router runs completed.")

    route_index = load_route_index(first_out)
    metadata["oos_start_date_actual"] = str(route_index.min().date())
    metadata["oos_end_date_actual"] = str(route_index.max().date())
    metadata["oos_observations"] = int(len(route_index))

    common_index = filtered_common_index(first_cfg, pair_results_dir)
    common_index = common_index[common_index <= route_index.max()]
    frames = load_filtered_pair_results(first_cfg, pair_results_dir, common_index)
    fixed_pair, fixed_returns = best_fixed_pair(frames, route_index)
    rows.append(
        metric_row(
            f"best_fixed_pair_ex_post:{fixed_pair}",
            "fixed_pair_reference",
            fixed_returns,
            fixed_pair,
            {"selected_pair": fixed_pair},
        )
    )
    save_timeseries(out_dir, "best_fixed_pair_ex_post", fixed_returns, fixed_pair)

    naive_path = Path(args.naive_scaling_path)
    if not naive_path.exists():
        raise FileNotFoundError(f"Naive scaling parquet not found: {naive_path}")
    naive = _normalize_pair_result(pd.read_parquet(naive_path))
    naive.index = pd.to_datetime(naive.index)
    naive_returns = pd.to_numeric(naive.reindex(route_index)["returns"], errors="coerce").fillna(0.0)
    rows.append(
        metric_row(
            "rv22_naive_scaling",
            "fixed_pair_benchmark",
            naive_returns,
            "rv22_naive_scaling",
            {"source": str(naive_path)},
        )
    )
    save_timeseries(out_dir, "rv22_naive_scaling", naive_returns, "rv22_naive_scaling")

    full_naive = naive.reindex(common_index)
    asset_returns = pd.to_numeric(full_naive["asset_returns"], errors="coerce").fillna(0.0)
    rf_returns = pd.to_numeric(full_naive["rf_daily_return"], errors="coerce").fillna(0.0)

    ewma_returns, ewma_weight = vol_target_returns(
        asset_returns=asset_returns,
        rf_returns=rf_returns,
        target_vol=0.10,
        mode="ewma",
        span=22,
        cap=1.5,
    )
    rows.append(
        metric_row(
            "ewma_volatility_targeting",
            "vol_target_benchmark",
            ewma_returns.loc[route_index],
            "ewma_volatility_targeting",
            {"avg_weight": float(ewma_weight.loc[route_index].mean())},
        )
    )
    save_timeseries(
        out_dir,
        "ewma_volatility_targeting",
        ewma_returns.loc[route_index],
        "ewma_volatility_targeting",
    )

    smooth_returns, smooth_weight = vol_target_returns(
        asset_returns=asset_returns,
        rf_returns=rf_returns,
        target_vol=0.10,
        mode="rv",
        window=22,
        cap=1.25,
        smooth=0.80,
    )
    rows.append(
        metric_row(
            "smoothed_capped_vol_targeting",
            "vol_target_benchmark",
            smooth_returns.loc[route_index],
            "smoothed_capped_vol_targeting",
            {"avg_weight": float(smooth_weight.loc[route_index].mean())},
        )
    )
    save_timeseries(
        out_dir,
        "smoothed_capped_vol_targeting",
        smooth_returns.loc[route_index],
        "smoothed_capped_vol_targeting",
    )

    regimes = load_regime_series(Path(args.regime_path), common_index)
    regime_returns, regime_selected, mapping = regime_aware_fixed_mapping(
        frames=frames,
        common_index=common_index,
        route_index=route_index,
        regimes=regimes,
        train_days=504,
    )
    rows.append(
        metric_row(
            "regime_aware_fixed_mapping",
            "regime_aware_fixed_mapping",
            regime_returns,
            regime_selected,
            {"mapping": json.dumps(mapping, sort_keys=True)},
        )
    )
    save_timeseries(out_dir, "regime_aware_fixed_mapping", regime_returns, regime_selected)

    table = pd.DataFrame(rows)
    ordered = [
        "strategy",
        "router_type",
        "ann_vol",
        "ann_return",
        "sharpe",
        "max_dd",
        "cvar",
        "switch_count",
        "unique_pair_use",
        "sortino",
        "calmar",
        "start",
        "end",
        "n_obs",
    ]
    extras = [col for col in table.columns if col not in ordered]
    table = table[ordered + extras]
    table.to_csv(out_dir / "non_ai_router_results.csv", index=False)
    table.to_parquet(out_dir / "non_ai_router_results.parquet", index=False)
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"wrote {out_dir / 'non_ai_router_results.csv'}")


if __name__ == "__main__":
    main()
