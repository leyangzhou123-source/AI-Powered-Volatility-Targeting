"""Optuna search for estimator params with prediction-error objective.

Objective:
    L(theta) = RMSE(pred, true) + lambda1 * Bias + lambda2 * TailPenalty
where:
    RMSE = sqrt(mean((pred-true)^2))
    Bias = abs(mean(pred-true))
    TailPenalty = RMSE on high-vol regime (true >= q80)
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.env import Env


def _resolve_yaml_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    p2 = Env.path("strategies") / p.name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"YAML not found: {path}")


def _load_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = _resolve_yaml_path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml: {p}")
    return data


def _load_data_from_cfg(cfg: dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg.get("data", {}) or {}
    data_path = data_cfg.get("path")
    if data_path is None:
        data_path = Env.path("processed") / "ES_Daily_Processed.parquet"
    else:
        data_path = Path(data_path)
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if "returns_clean" not in df.columns:
        raise ValueError("Data must include 'returns_clean'")
    return df


def _realized_vol_next(returns: pd.Series, window: int = 21, ann: float = 252.0) -> pd.Series:
    r = pd.Series(returns).astype(float)
    # Forward realized volatility proxy from future squared returns.
    fwd_r2 = (r.shift(-1) ** 2).rolling(window, min_periods=window).sum()
    rv = np.sqrt((ann / window) * fwd_r2)
    return rv.rename("sigma_true")


def _sample_param(trial, key: str, spec: dict[str, Any]):
    t = str(spec.get("type", "float"))
    if t == "int":
        return trial.suggest_int(key, int(spec["low"]), int(spec["high"]))
    if t == "float":
        return trial.suggest_float(key, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
    if t == "categorical":
        return trial.suggest_categorical(key, list(spec["choices"]))
    raise ValueError(f"Unknown spec type for {key}: {t}")


def _rolling_pred(estimator, returns: pd.Series, roll_window: int) -> pd.Series:
    idx = returns.index
    out = np.full(len(idx), np.nan, dtype=float)

    for i in range(roll_window, len(idx)):
        t = idx[i]
        w = returns.iloc[i - roll_window : i].dropna().astype(float)
        if len(w) == 0:
            continue
        try:
            if hasattr(estimator, "estimate_window"):
                v = float(estimator.estimate_window(w))
            else:
                v = float(estimator.estimate(t, w))
            if np.isfinite(v) and v > 0:
                out[i] = v
        except Exception:
            continue

    return pd.Series(out, index=idx, name="sigma_hat")


def _metrics(pred: pd.Series, true: pd.Series) -> dict[str, float]:
    m = pd.concat([pred.rename("pred"), true.rename("true")], axis=1).dropna()
    if len(m) < 30:
        return {"rmse": np.inf, "bias": np.inf, "tail": np.inf}

    err = (m["pred"] - m["true"]).to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(abs(np.mean(err)))

    q = float(m["true"].quantile(0.8))
    tail_mask = m["true"] >= q
    if tail_mask.sum() < 5:
        tail = rmse
    else:
        terr = (m.loc[tail_mask, "pred"] - m.loc[tail_mask, "true"]).to_numpy(dtype=float)
        tail = float(np.sqrt(np.mean(terr**2)))

    return {"rmse": rmse, "bias": bias, "tail": tail}


def main():
    parser = argparse.ArgumentParser(description="Optuna estimator search by prediction error objective")
    parser.add_argument("--strategy", "-s", required=True, help="Strategy YAML path")
    parser.add_argument("--pair-name", default=None, help="router pair name to tune (default: router.default_pair or first)")
    parser.add_argument("--search-space", default="configs/optuna/search_space.yaml")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--lambda1", type=float, default=0.5, help="bias penalty weight")
    parser.add_argument("--lambda2", type=float, default=1.0, help="tail penalty weight")
    parser.add_argument("--roll-window", type=int, default=252)
    parser.add_argument("--true-window", type=int, default=21)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(Path("results") / "evaluation" / "optuna"))
    parser.add_argument("--write-back", action="store_true", help="Write best estimator params back to strategy YAML")
    parser.add_argument(
        "--write-all-matching-pairs",
        action="store_true",
        help="With --write-back, apply best params to all pairs that share the same estimator class",
    )
    parser.add_argument("--backup-suffix", default=".bak", help="Backup suffix used with --write-back")
    args = parser.parse_args()

    try:
        import optuna
    except Exception as e:
        raise RuntimeError("optuna is required: pip install optuna") from e

    strategy_path = _resolve_yaml_path(args.strategy)
    cfg = _load_yaml(strategy_path)
    space = _load_yaml(args.search_space)
    df = _load_data_from_cfg(cfg)

    router = cfg.get("router", {}) or {}
    pairs = list(router.get("pairs", []) or [])
    if not pairs:
        raise ValueError("Strategy must include router.pairs")

    target_pair = None
    if args.pair_name:
        for p in pairs:
            if p.get("name") == args.pair_name:
                target_pair = p
                break
    if target_pair is None:
        default = router.get("default_pair")
        if default:
            for p in pairs:
                if p.get("name") == default:
                    target_pair = p
                    break
    if target_pair is None:
        target_pair = pairs[0]

    est_cfg = copy.deepcopy(target_pair.get("estimator", {}) or {})
    est_path = est_cfg.get("class")
    if not est_path:
        raise ValueError("Selected pair missing estimator.class")

    base_params = copy.deepcopy(est_cfg.get("params", {}) or {})
    tunable_keys = [k for k in base_params.keys() if k in space]
    if not tunable_keys:
        raise ValueError("No tunable estimator params found in search space for selected pair")

    split = int(len(df) * float(args.train_ratio))
    returns = df["returns_clean"].astype(float)
    train_ret = returns.iloc[:split]
    valid_ret = returns.iloc[max(0, split - args.roll_window - 5) :]

    true_valid = _realized_vol_next(valid_ret, window=args.true_window, ann=252.0)

    EstCls = _load_class(est_path)

    trial_rows: list[dict[str, Any]] = []

    def objective(trial):
        params = copy.deepcopy(base_params)
        for k in tunable_keys:
            params[k] = _sample_param(trial, k, space[k])

        estimator = EstCls(params)
        pred = _rolling_pred(estimator, valid_ret, roll_window=args.roll_window)

        m = _metrics(pred, true_valid)
        loss = float(m["rmse"] + args.lambda1 * m["bias"] + args.lambda2 * m["tail"])

        trial.set_user_attr("rmse", m["rmse"])
        trial.set_user_attr("bias", m["bias"])
        trial.set_user_attr("tail", m["tail"])

        trial_rows.append(
            {
                "trial": trial.number,
                "loss": loss,
                "rmse": m["rmse"],
                "bias": m["bias"],
                "tail": m["tail"],
                "params": json.dumps(params, ensure_ascii=True),
            }
        )
        return loss

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.strategy).stem}__{target_pair.get('name','pair')}"

    best = {
        "strategy": str(strategy_path),
        "pair": target_pair.get("name"),
        "estimator_class": est_path,
        "objective": "rmse + lambda1*bias + lambda2*tail",
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
    }

    pd.DataFrame(trial_rows).sort_values("loss").to_csv(out_dir / f"{stem}_trials.csv", index=False)
    with open(out_dir / f"{stem}_best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=True)

    if args.write_back:
        cfg_out = copy.deepcopy(cfg)
        pairs_out = list((cfg_out.get("router", {}) or {}).get("pairs", []) or [])
        target_name = target_pair.get("name")
        updated_pairs: list[str] = []
        for p in pairs_out:
            est = p.get("estimator", {}) or {}
            est_cls = est.get("class")

            should_update = False
            if args.write_all_matching_pairs:
                should_update = est_cls == est_path
            else:
                should_update = p.get("name") == target_name

            if should_update:
                cur = est.get("params", {}) or {}
                cur.update(study.best_params)
                est["params"] = cur
                p["estimator"] = est
                updated_pairs.append(str(p.get("name")))

        if not updated_pairs:
            if args.write_all_matching_pairs:
                raise RuntimeError(f"Could not find any pairs matching estimator class '{est_path}'.")
            raise RuntimeError(f"Could not find pair '{target_name}' for write-back.")

        backup_path = Path(str(strategy_path) + args.backup_suffix)
        backup_path.write_text(Path(strategy_path).read_text(encoding="utf-8"), encoding="utf-8")
        Path(strategy_path).write_text(yaml.safe_dump(cfg_out, sort_keys=False, allow_unicode=False), encoding="utf-8")

    print("=" * 60)
    print(f"Search done: {stem}")
    print(f"Best loss: {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Saved: {out_dir / (stem + '_trials.csv')}")
    print(f"Saved: {out_dir / (stem + '_best.json')}")
    if args.write_back:
        print(f"Backed up: {backup_path}")
        print(f"Updated strategy: {strategy_path}")
        print(f"Updated pairs ({len(updated_pairs)}): {', '.join(updated_pairs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
