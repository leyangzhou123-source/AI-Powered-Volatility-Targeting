"""
run_friction_tests.py
─────────────────────────────────────────────────────────────────────────────
Transaction cost sensitivity analysis, split by volatility regime.

Generates 4 PNGs in results/:
  1. 6_Friction_Overall_<Controller>.png      — full-sample Sharpe vs cost
  2. 6_Friction_Regime_Low_<Controller>.png   — Sharpe computed on Low-vol days only
  3. 6_Friction_Regime_Mid_<Controller>.png   — Sharpe computed on Mid-vol days only
  4. 6_Friction_Regime_High_<Controller>.png  — Sharpe computed on High-vol days only

Regimes are defined ONCE from actual realised volatility (RV20) via a
3-state Gaussian HMM — identical to analyze_regimes.py and
plot_vol_regime_timeseries.py — so all charts share the same baseline.

Usage
─────
    python scripts/run_friction_tests.py --controller NaiveScaling
    python scripts/run_friction_tests.py --controller RegimeSwitchController
    python scripts/run_friction_tests.py --controller VarianceScaling
"""

import argparse
import sys
import yaml
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import Env
from src.backtest import VolTargetEngine


# ── constants ─────────────────────────────────────────────────────────────────
COST_TIERS   = [0.0, 2.0, 5.0, 10.0, 15.0]
REGIME_ORDER = ["Low", "Mid", "High"]
REGIME_COLORS = {
    "Low":  "#2ecc71",
    "Mid":  "#f39c12",
    "High": "#e74c3c",
}
MARKERS = ["o", "s", "^", "D", "v", "p", "*", "X", "h"]


# ── global regime fitting (same as analyze_regimes.py) ───────────────────────
def fit_global_regimes() -> pd.Series:
    """
    Fit a 3-state Gaussian HMM on log(rv20) from prices.parquet.
    Returns a Series indexed by date with values 'Low' | 'Mid' | 'High'.
    """
    print("🌍  Fitting global HMM on RV20...")

    prices_path = Env.path("prices", "databento") / "prices.parquet"
    if not prices_path.exists():
        prices_path = Env.path("raw") / "prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(
            f"prices.parquet not found at {prices_path}. Run run_importers.py first."
        )

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(subset=["rv20"]).sort_index()

    rv20   = prices["rv20"].astype(float)
    log_rv = np.log(rv20 + 1e-8).values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=3, covariance_type="diag",
        n_iter=1000, random_state=42,
    )
    model.fit(log_rv)
    states = model.predict(log_rv)

    order   = np.argsort(model.means_.flatten())
    mapping = {int(order[0]): "Low", int(order[1]): "Mid", int(order[2]): "High"}
    regime  = pd.Series(states, index=prices.index).map(mapping)
    regime.name = "global_regime"

    for r in REGIME_ORDER:
        print(f"   {r:4s}: {(regime == r).sum():,} days ({(regime == r).mean():.1%})")

    return regime


# ── yaml helpers ──────────────────────────────────────────────────────────────
def _class_name(class_path: str) -> str:
    return class_path.rsplit(".", 1)[-1]


def load_configs_for_controller(strategies_dir: Path, controller_name: str) -> list[dict]:
    matched = []
    for p in strategies_dir.glob("*.yaml"):
        try:
            with open(p, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not cfg:
                continue
            if _class_name(cfg.get("controller", {}).get("class", "")) == controller_name:
                matched.append(cfg)
        except Exception as exc:
            print(f"  ⚠️  Could not load {p.name}: {exc}")

    if not matched:
        # show available controllers to help the user
        available = set()
        for p in strategies_dir.glob("*.yaml"):
            try:
                with open(p, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if cfg:
                    available.add(_class_name(cfg.get("controller", {}).get("class", "")))
            except Exception:
                pass
        raise ValueError(
            f"No YAML configs found for controller '{controller_name}'.\n"
            f"Available controllers: {sorted(available)}"
        )

    return matched


# ── sharpe helper ─────────────────────────────────────────────────────────────
def _sharpe(returns: pd.Series) -> float:
    rets = returns.fillna(0)
    if len(rets) < 5:
        return np.nan
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    return ann_ret / ann_vol if ann_vol > 0 else np.nan


# ── core sensitivity runner ───────────────────────────────────────────────────
def run_sensitivity(
    configs: list[dict],
    global_regime: pd.Series,
) -> dict:
    """
    For every (strategy, cost_tier) combination run the engine and collect:
      - overall Sharpe
      - Sharpe on Low / Mid / High regime days only

    Returns nested dict:
        data[strat_name][cost] = {
            "Overall": float,
            "Low": float,
            "Mid": float,
            "High": float,
        }
    """
    data = {}

    for cfg in configs:
        strat_name = cfg["name"]
        print(f"\n  ▶ {strat_name}")
        data[strat_name] = {}

        for cost in COST_TIERS:
            test_cfg = copy.deepcopy(cfg)
            test_cfg["cost_bps"] = cost

            try:
                engine = VolTargetEngine.from_config(test_cfg)
                res    = engine.run(mode="all")
            except Exception as exc:
                print(f"    ⚠️  cost={cost} bps failed: {exc}")
                data[strat_name][cost] = {r: np.nan for r in ["Overall"] + REGIME_ORDER}
                continue

            rets = res["returns"]

            # Overall Sharpe
            overall = _sharpe(rets)

            # Regime-split Sharpe — join on global regime index
            regime_aligned = global_regime.reindex(rets.index)
            regime_sharpes = {}
            for r in REGIME_ORDER:
                mask = regime_aligned == r
                regime_sharpes[r] = _sharpe(rets[mask]) if mask.sum() > 5 else np.nan

            data[strat_name][cost] = {"Overall": overall, **regime_sharpes}
            print(
                f"    cost={cost:5.1f} bps | "
                f"Overall={overall:+.3f} | "
                + " | ".join(
                    f"{r}={regime_sharpes[r]:+.3f}"
                    if np.isfinite(regime_sharpes[r]) else f"{r}=—"
                    for r in REGIME_ORDER
                )
            )

    return data


# ── plotting ──────────────────────────────────────────────────────────────────
def _plot_panel(
    sensitivity_data: dict,
    slice_key: str,          # "Overall" | "Low" | "Mid" | "High"
    controller_name: str,
    results_dir: Path,
) -> None:
    """Draw and save one Sharpe-vs-cost chart for the given slice."""

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(13, 7))

    for i, (strat_name, cost_map) in enumerate(sensitivity_data.items()):
        sharpes = [cost_map[c].get(slice_key, np.nan) for c in COST_TIERS]
        marker  = MARKERS[i % len(MARKERS)]

        # Use regime colour for regime panels, default cycle for overall
        color = REGIME_COLORS.get(slice_key, None)
        kwargs = dict(marker=marker, linewidth=2, markersize=7, label=strat_name)
        if color:
            kwargs["color"] = color

        ax.plot(COST_TIERS, sharpes, **kwargs)

    # Zero-Sharpe reference line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    if slice_key == "Overall":
        title    = f"Transaction Cost Sensitivity — All Periods\n(Controller: {controller_name})"
        filename = f"6_Friction_Overall_{controller_name}.png"
    else:
        regime_label = {"Low": "Low Vol", "Mid": "Mid Vol", "High": "High Vol"}[slice_key]
        title    = (
            f"Transaction Cost Sensitivity — {regime_label} Regime\n"
            f"(Controller: {controller_name}  ·  Regimes from RV20 HMM)"
        )
        filename = f"6_Friction_Regime_{slice_key}_{controller_name}.png"

        # Light background tint matching the regime colour
        ax.set_facecolor(REGIME_COLORS[slice_key] + "18")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Transaction Costs (Basis Points)", fontsize=12)
    ax.set_ylabel("Annualised Sharpe Ratio", fontsize=12)
    ax.set_xticks(COST_TIERS)
    ax.legend(
        title="Estimator",
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=9, title_fontsize=10, framealpha=0.9,
    )
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = results_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved → {filename}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Friction sensitivity analysis split by RV20 volatility regime."
    )
    parser.add_argument(
        "--controller", "-c",
        required=True,
        help="Controller class name, e.g. NaiveScaling | RegimeSwitchController | VarianceScaling",
    )
    args = parser.parse_args()
    controller_name = args.controller.strip()

    results_dir    = Env.path("results")
    strategies_dir = Env.path("strategies")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Fit global regimes once ────────────────────────────────────────────
    global_regime = fit_global_regimes()

    # ── 2. Load configs for this controller ───────────────────────────────────
    configs = load_configs_for_controller(strategies_dir, controller_name)
    print(
        f"\n🎯  Controller : {controller_name}"
        f"\n📂  Strategies : {[c['name'] for c in configs]}\n"
        f"{'─'*60}"
    )

    # ── 3. Run sensitivity across all cost tiers ──────────────────────────────
    print("\n🚀  Running transaction cost sensitivity tests...")
    sensitivity_data = run_sensitivity(configs, global_regime)

    # ── 4. Generate the 4 plots ───────────────────────────────────────────────
    print(f"\n📊  Generating plots...")
    for slice_key in ["Overall"] + REGIME_ORDER:
        _plot_panel(sensitivity_data, slice_key, controller_name, results_dir)

    print(f"\n{'─'*60}")
    print(f"✅  Done. 4 plots saved to: {results_dir}")


if __name__ == "__main__":
    main()