"""Build crypto vol pairs and evaluate them with the AI-regime router protocol."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.router.evaluate_router_protocol_precomputed import evaluate_precomputed_router_protocol  # noqa: E402
from src.backtest import VolTargetEngine  # noqa: E402
from src.env import Env  # noqa: E402


ASSETS: dict[str, dict[str, Any]] = {
    "btc": {
        "symbol": "BTCUSD",
        "daily_path": "data/processed/BTCUSD_daily.parquet",
        "intraday_path": "data/processed/BTCUSD_intraday_volatility.parquet",
        "target_vol": 0.35,
        "weight_max": 1.25,
        "cost_bps": 8.0,
    },
    "usdt": {
        "symbol": "USDTUSD",
        "daily_path": "data/processed/USDTUSD_daily.parquet",
        "intraday_path": "data/processed/USDTUSD_intraday_volatility.parquet",
        "target_vol": 0.02,
        "weight_max": 2.0,
        "cost_bps": 2.0,
    },
}


def _redacted_router_config(cfg: dict[str, Any], api_key_env: str = "NVAPI_KEY") -> dict[str, Any]:
    disk_cfg = copy.deepcopy(cfg)
    params = disk_cfg.get("router", {}).get("params", {})
    if "api_key" in params:
        params["api_key"] = f"${{{api_key_env}}}"
    return disk_cfg


ESTIMATORS: list[dict[str, Any]] = [
    {
        "slug": "crypto_intraday_rv_1d",
        "class": "src.crypto_router.estimators.IntradayRealizedVolEstimator",
        "params": {"lookback": 1, "fallback_lookback": 20, "vol_ann": 365},
    },
    {
        "slug": "crypto_intraday_rv_3d",
        "class": "src.crypto_router.estimators.IntradayRealizedVolEstimator",
        "params": {"lookback": 3, "fallback_lookback": 20, "vol_ann": 365},
    },
    {
        "slug": "crypto_intraday_rv_7d",
        "class": "src.crypto_router.estimators.IntradayRealizedVolEstimator",
        "params": {"lookback": 7, "fallback_lookback": 30, "vol_ann": 365},
    },
    {
        "slug": "crypto_range_gk_20d",
        "class": "src.crypto_router.estimators.RangeBasedVolEstimator",
        "params": {"lookback": 20, "method": "garman_klass", "vol_ann": 365},
    },
    {
        "slug": "crypto_range_gk_60d",
        "class": "src.crypto_router.estimators.RangeBasedVolEstimator",
        "params": {"lookback": 60, "method": "garman_klass", "vol_ann": 365},
    },
    {
        "slug": "crypto_range_parkinson_20d",
        "class": "src.crypto_router.estimators.RangeBasedVolEstimator",
        "params": {"lookback": 20, "method": "parkinson", "vol_ann": 365},
    },
    {
        "slug": "crypto_composite_fast",
        "class": "src.crypto_router.estimators.CryptoCompositeVolEstimator",
        "params": {
            "lookback": 10,
            "ewma_halflife": 5,
            "weights": [0.60, 0.20, 0.20],
            "vol_ann": 365,
        },
    },
    {
        "slug": "crypto_composite_balanced",
        "class": "src.crypto_router.estimators.CryptoCompositeVolEstimator",
        "params": {
            "lookback": 20,
            "ewma_halflife": 10,
            "weights": [0.50, 0.25, 0.25],
            "vol_ann": 365,
        },
    },
    {
        "slug": "crypto_composite_slow",
        "class": "src.crypto_router.estimators.CryptoCompositeVolEstimator",
        "params": {
            "lookback": 60,
            "ewma_halflife": 20,
            "weights": [0.35, 0.35, 0.30],
            "vol_ann": 365,
        },
    },
]

CONTROLLERS: list[dict[str, Any]] = [
    {
        "slug": "naive_scaling",
        "class": "src.controllers.naive_scaling.NaiveScaling",
        "params": {},
    },
    {
        "slug": "hysteresis",
        "class": "src.controllers.hysteresis_controller.HysteresisController",
        "params": {"deadband": 0.025, "w_min": 0.0, "w_max": 1.5},
    },
    {
        "slug": "vol_target_clip",
        "class": "src.controllers.vol_target_clip.VolTargetClip",
        "params": {"w_min": 0.0, "w_max": 1.5, "rebalance_threshold": 0.025},
    },
    {
        "slug": "variance_scaling",
        "class": "src.controllers.variance_scaling.VarianceScaling",
        "params": {"sigma_bar_window": 90, "no_trade_band": 0.025},
    },
    {
        "slug": "drawdown_brake",
        "class": "src.controllers.drawdown_brake.DrawdownBrake",
        "params": {},
    },
    {
        "slug": "drawdown_modulated",
        "class": "src.controllers.drawdown_modulated.DrawdownModulatedController",
        "params": {},
    },
    {
        "slug": "priority_stack",
        "class": "src.controllers.priority_stack_controller.PriorityStackController",
        "params": {},
    },
    {
        "slug": "regime_switch",
        "class": "src.controllers.regime_controller.RegimeSwitchController",
        "params": {},
    },
    {
        "slug": "trend_filter",
        "class": "src.controllers.trend_filter.TrendFilter",
        "params": {},
    },
    {
        "slug": "vol_shock_throttle",
        "class": "src.crypto_router.controllers.VolatilityShockThrottle",
        "params": {
            "no_trade_band": 0.025,
            "shock_multiplier": 1.75,
            "shock_cut": 0.50,
            "max_step_up": 0.20,
        },
    },
    {
        "slug": "peg_aware_vol",
        "class": "src.crypto_router.controllers.PegAwareVolController",
        "params": {
            "no_trade_band": 0.01,
            "peg_deviation_soft": 0.0015,
            "peg_deviation_hard": 0.0060,
            "max_step": 0.25,
        },
    },
]

BASELINE_PAIR = {
    "name_suffix": "realized_vol_20d__naive_scaling",
    "estimator": {
        "class": "src.estimators.realized_vol.RealizedVol",
        "params": {"lookback": 20, "vol_ann": 365},
    },
    "controller": {"class": "src.controllers.naive_scaling.NaiveScaling", "params": {}},
}


def load_env_file(path: str | Path | None) -> None:
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


def _asset_cfg(asset: str) -> dict[str, Any]:
    key = asset.lower()
    if key not in ASSETS:
        raise ValueError(f"Unknown asset {asset!r}; expected one of {sorted(ASSETS)}")
    return dict(ASSETS[key])


def _pair_cfgs(asset: str, asset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for estimator in ESTIMATORS:
        for controller in CONTROLLERS:
            est_params = dict(estimator["params"])
            est_params["path"] = asset_cfg["intraday_path"]
            ctrl_params = dict(controller["params"])
            if "w_max" in ctrl_params:
                ctrl_params["w_max"] = asset_cfg["weight_max"]
            name = f"{asset}_{estimator['slug']}__{controller['slug']}"
            out.append(
                {
                    "name": name,
                    "estimator": {"class": estimator["class"], "params": est_params},
                    "controller": {"class": controller["class"], "params": ctrl_params},
                }
            )
    return out


def _baseline_pair_cfg(asset: str) -> dict[str, Any]:
    return {
        "name": f"{asset}_{BASELINE_PAIR['name_suffix']}",
        "estimator": {
            "class": BASELINE_PAIR["estimator"]["class"],
            "params": dict(BASELINE_PAIR["estimator"]["params"]),
        },
        "controller": {
            "class": BASELINE_PAIR["controller"]["class"],
            "params": dict(BASELINE_PAIR["controller"]["params"]),
        },
    }


def _base_pair_config(asset: str, asset_cfg: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": pair["name"],
        "description": f"{asset_cfg['symbol']} crypto volatility estimator/controller pair",
        "data": {
            "symbol": asset_cfg["symbol"],
            "path": asset_cfg["daily_path"],
            "date_col": "date",
            "returns_col": "log_return",
            "price_col": "close",
        },
        "intraday_realized_vol": {
            "enabled": True,
            "path": asset_cfg["intraday_path"],
            "lookback": 21,
        },
        "target_vol": asset_cfg["target_vol"],
        "rebalance_freq": "daily",
        "roll_window": 90,
        "weight_min": 0.0,
        "weight_max": asset_cfg["weight_max"],
        "cost_bps": asset_cfg["cost_bps"],
        "vol_ann": 365,
        "router": {"enabled": False},
        "estimator": pair["estimator"],
        "controller": pair["controller"],
    }


def build_pair_results(asset: str, out_dir: Path, mode: str, resume: bool) -> pd.DataFrame:
    asset_cfg = _asset_cfg(asset)
    pair_cfgs = [_baseline_pair_cfg(asset), *_pair_cfgs(asset, asset_cfg)]
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for pair in pair_cfgs:
        path = out_dir / f"{pair['name']}.parquet"
        if resume and path.exists():
            rows = int(len(pd.read_parquet(path)))
            records.append({"name": pair["name"], "status": "ok", "rows": rows, "path": str(path)})
            continue
        cfg = _base_pair_config(asset, asset_cfg, pair)
        result = VolTargetEngine.from_config(cfg).run(mode=mode)
        result.to_parquet(path)
        records.append({"name": pair["name"], "status": "ok", "rows": int(len(result)), "path": str(path)})
        print(f"[crypto-pairs] wrote {path} rows={len(result)}", flush=True)
    manifest = pd.DataFrame(records)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def build_router_config(
    asset: str,
    pair_results_dir: Path,
    config_path: Path,
    precomputed_ai_regime_path: str,
    max_ai_calls: int,
    model: str,
    provider: str,
    api_key: str,
    timeout: float,
    max_output_tokens: int,
    switch_decision_max_output_tokens: int,
    selection_max_output_tokens: int,
    request_min_interval_seconds: float,
    api_key_env: str = "NVAPI_KEY",
) -> dict[str, Any]:
    asset_cfg = _asset_cfg(asset)
    pairs = _pair_cfgs(asset, asset_cfg)
    benchmark_path = pair_results_dir / f"{asset}_{BASELINE_PAIR['name_suffix']}.parquet"
    cfg = {
        "name": f"{asset}_crypto_ai_regime_router",
        "description": f"AI-regime router over {asset_cfg['symbol']} crypto volatility pairs",
        "data": {
            "symbol": asset_cfg["symbol"],
            "path": asset_cfg["daily_path"],
            "date_col": "date",
            "returns_col": "log_return",
            "price_col": "close",
        },
        "intraday_realized_vol": {
            "enabled": True,
            "path": asset_cfg["intraday_path"],
            "lookback": 21,
        },
        "pair_history_features": {
            "enabled": True,
            "path": str(pair_results_dir),
            "benchmark_path": str(benchmark_path),
            "lookbacks": [10, 30, 90],
        },
        "target_vol": asset_cfg["target_vol"],
        "rebalance_freq": "daily",
        "roll_window": 90,
        "weight_min": 0.0,
        "weight_max": asset_cfg["weight_max"],
        "cost_bps": asset_cfg["cost_bps"],
        "vol_ann": 365,
        "router": {
            "enabled": True,
            "type": "ai_regime",
            "class": "src.router.ai_regime_router.AIRegimeRouter",
            "params": {
                "provider": provider,
                "model": model,
                "api_format": "chat_completions",
                "api_key": api_key,
                "ai_enabled": max_ai_calls != 0,
                "fail_open": True,
                "precomputed_ai_regime_path": precomputed_ai_regime_path,
                "precomputed_ai_regime_only": bool(precomputed_ai_regime_path),
                "max_ai_regime_calls": max_ai_calls,
                "max_ai_selection_calls": max_ai_calls,
                "max_total_ai_calls": max_ai_calls,
                "ai_regime_interval": 10,
                "ai_selection_interval": 10,
                "candidate_top_n": 6,
                "sensitiveness": "normal",
                "sticky_period": 1,
                "perf_weight": 1.5,
                "lambda_drawdown": 0.5,
                "lambda_switch": 0.15,
                "lambda_invalid": 2.0,
                "lambda_exception": 1.0,
                "use_heuristic_regime_bias": False,
                "regime_bias_weight": 3.0,
                "max_output_tokens": max_output_tokens,
                "switch_decision_max_output_tokens": switch_decision_max_output_tokens,
                "selection_max_output_tokens": selection_max_output_tokens,
                "timeout": timeout,
                "request_min_interval_seconds": request_min_interval_seconds,
            },
            "pairs": pairs,
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(_redacted_router_config(cfg, api_key_env), sort_keys=False), encoding="utf-8")
    return cfg


def _ann_vol(returns: pd.Series, ann: float = 365.0) -> float:
    r = pd.to_numeric(returns, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * (float(ann) ** 0.5))


def _sharpe(returns: pd.Series, ann: float = 365.0) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    vol = _ann_vol(r, ann=ann)
    return float(r.mean() * float(ann) / vol) if vol > 0 else 0.0


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = pd.Series(np.exp(r.cumsum()), index=r.index)
    dd = equity / equity.cummax() - 1.0
    return float(-dd.min()) if len(dd) else 0.0


def write_baseline_comparison(output_dir: Path, baseline_path: Path, ann: float = 365.0) -> Path | None:
    ts_path = output_dir / "router_vs_bh_timeseries.csv"
    if not ts_path.exists() or not baseline_path.exists():
        return None
    ts = pd.read_csv(ts_path, parse_dates=["date"])
    baseline = pd.read_parquet(baseline_path)
    baseline.index = pd.to_datetime(baseline.index)
    baseline_returns = (
        baseline["returns_with_rf"]
        if "returns_with_rf" in baseline.columns
        else baseline.get("returns", baseline.get("returns_no_rf"))
    )
    aligned = ts.set_index("date").join(baseline_returns.rename("baseline_return"), how="inner")
    router_r = aligned["router_return"]
    baseline_r = aligned["baseline_return"]
    rows = [
        {
            "strategy": "ai_regime_combined",
            "n_obs": int(router_r.notna().sum()),
            "ann_return": float(router_r.mean() * float(ann)),
            "ann_vol": _ann_vol(router_r, ann=ann),
            "sharpe": _sharpe(router_r, ann=ann),
            "max_drawdown": _max_drawdown_from_returns(router_r),
            "ann_factor": float(ann),
        },
        {
            "strategy": "realized_vol_20d__naive_scaling",
            "n_obs": int(baseline_r.notna().sum()),
            "ann_return": float(baseline_r.mean() * float(ann)),
            "ann_vol": _ann_vol(baseline_r, ann=ann),
            "sharpe": _sharpe(baseline_r, ann=ann),
            "max_drawdown": _max_drawdown_from_returns(baseline_r),
            "ann_factor": float(ann),
        },
    ]
    out = output_dir / "router_vs_baseline_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def run_router(asset: str, pair_results_dir: Path, output_dir: Path, cfg: dict[str, Any], args) -> dict[str, pd.DataFrame]:
    return evaluate_precomputed_router_protocol(
        config_path=str(args.config_out),
        pair_results_dir=pair_results_dir,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        max_pairs=args.max_pairs,
        output_dir=output_dir,
        config=cfg,
        oos_start_date=args.oos_start_date,
        oos_end_date=args.oos_end_date,
        single_oos_window=True,
        freeze_oos_metrics=args.freeze_oos_metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BTC/USDT crypto vol pairs and run AI-regime router.")
    parser.add_argument("--asset", choices=sorted(ASSETS), default="btc")
    parser.add_argument("--mode", choices=["all", "in_sample", "out_of_sample"], default="all")
    parser.add_argument("--pair-results-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--config-out", type=Path, default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--precomputed-ai-regime-path", default="")
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--api-key-env", default="NVAPI_KEY")
    parser.add_argument("--max-ai-calls", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--switch-decision-max-output-tokens", type=int, default=2048)
    parser.add_argument("--selection-max-output-tokens", type=int, default=2048)
    parser.add_argument("--request-min-interval-seconds", type=float, default=0.0)
    parser.add_argument("--vol-target-ai-router", action="store_true")
    parser.add_argument("--train-candidate-pool-size", type=int, default=30)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=180)
    parser.add_argument("--step-days", type=int, default=180)
    parser.add_argument("--oos-start-date", default="2023-01-01")
    parser.add_argument("--oos-end-date", default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--freeze-oos-metrics", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    load_env_file(args.env_file)

    asset = args.asset.lower()
    pair_results_dir = Path(args.pair_results_dir or Env.path("results") / f"crypto_{asset}_vol_pairs")
    output_dir = Path(args.output_dir or Env.path("evaluation") / f"crypto_{asset}_ai_regime_router")
    args.config_out = args.config_out or Env.path("strategies") / f"crypto_{asset}_router.yaml"
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing API key: {args.api_key_env} was not found after loading {args.env_file}")

    manifest = build_pair_results(asset, pair_results_dir, mode=args.mode, resume=not args.no_resume)
    cfg = build_router_config(
        asset=asset,
        pair_results_dir=pair_results_dir,
        config_path=args.config_out,
        precomputed_ai_regime_path=args.precomputed_ai_regime_path,
        max_ai_calls=args.max_ai_calls,
        model=args.model,
        provider=args.provider,
        api_key=api_key,
        timeout=args.timeout,
        max_output_tokens=args.max_output_tokens,
        switch_decision_max_output_tokens=args.switch_decision_max_output_tokens,
        selection_max_output_tokens=args.selection_max_output_tokens,
        request_min_interval_seconds=args.request_min_interval_seconds,
        api_key_env=args.api_key_env,
    )
    if args.vol_target_ai_router:
        router = cfg.setdefault("router", {})
        router["class"] = "src.crypto_router.ai_vol_target_router.CryptoVolTargetAIRouter"
        params = router.setdefault("params", {})
        params.update(
            {
                "target_vol": float(cfg.get("target_vol", _asset_cfg(asset)["target_vol"])),
                "ann_factor": float(cfg.get("vol_ann", 365)),
                "asset_label": asset.upper(),
                "train_candidate_filter_enabled": True,
                "train_candidate_pool_size": args.train_candidate_pool_size,
                "initial_pair_rule": "train_shape",
                "deterministic_switch_decision": False,
                "deterministic_pair_selection": False,
                "sensitiveness": "high",
                "sticky_period": 20,
                "ai_regime_interval": 20,
                "ai_selection_interval": 20,
                "candidate_top_n": 10,
            }
        )
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(
            yaml.safe_dump(_redacted_router_config(cfg, args.api_key_env), sort_keys=False),
            encoding="utf-8",
        )
    tables = run_router(asset, pair_results_dir, output_dir, cfg, args)
    baseline_path = pair_results_dir / f"{asset}_{BASELINE_PAIR['name_suffix']}.parquet"
    comparison_path = write_baseline_comparison(output_dir, baseline_path, ann=float(cfg.get("vol_ann", 365)))
    payload = {
        "asset": asset,
        "pair_results_dir": str(pair_results_dir),
        "config": str(args.config_out),
        "output_dir": str(output_dir),
        "pairs": int((manifest["status"] == "ok").sum()),
        "routed_pairs": len(_pair_cfgs(asset, _asset_cfg(asset))),
        "baseline": str(baseline_path),
        "baseline_comparison": str(comparison_path) if comparison_path else "",
        "summary": tables["summary"].to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
