

import argparse
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from hmmlearn.hmm import GaussianHMM


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import Env  


# ── constants ───────────────────────────────────────────────
REGIME_COLORS = {
    "Low":  "#2ecc71",   
    "Mid":  "#f39c12",   
    "High": "#e74c3c",   
}
REGIME_LABELS = {
    "Low":  "Low Vol",
    "Mid":  "Mid Vol",
    "High": "High Vol",
}
REGIME_ORDER = ["Low", "Mid", "High"]


# ── global regime fitting
def fit_global_regimes() -> pd.DataFrame:
    """
    Fit a 3-state Gaussian HMM on log(rv20) from prices.parquet."""


    prices_path = Env.path("prices", "databento") / "prices.parquet"
    if not prices_path.exists():
        prices_path = Env.path("raw") / "prices.parquet"

    if not prices_path.exists():
        raise FileNotFoundError(
            f"prices.parquet not found at {prices_path}. "
            "Run run_importers.py first."
        )

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(subset=["rv20"]).sort_index()

    rv20    = prices["rv20"].astype(float)
    log_rv  = np.log(rv20 + 1e-8).values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=3, covariance_type="diag",
        n_iter=1000, random_state=42,
    )
    model.fit(log_rv)
    states = model.predict(log_rv)

    #0=Low, 1=Mid, 2=High
    order   = np.argsort(model.means_.flatten())
    mapping = {int(order[0]): "Low", int(order[1]): "Mid", int(order[2]): "High"}

    regime_labels = pd.Series(states, index=prices.index).map(mapping)

    print(
        f"   HMM fitted on {len(rv20):,} days "
        + "  ".join(
            f"{REGIME_LABELS[r]}: {(regime_labels == r).sum():,}d "
            f"({(regime_labels == r).mean():.1%})"
            for r in REGIME_ORDER
        )
    )

    return pd.DataFrame({"rv20": rv20, "global_regime": regime_labels})


def _class_name(class_path: str) -> str:
    return class_path.rsplit(".", 1)[-1]


def load_yaml_configs(strategies_dir: Path) -> list[dict]:
    configs = []
    for p in strategies_dir.glob("*.yaml"):
        try:
            with open(p, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg:
                cfg["_yaml_stem"] = p.stem
                configs.append(cfg)
        except Exception as exc:
            print(f" Could not load {p.name}: {exc}")
    return configs


def filter_by_controller(configs: list[dict], controller_name: str) -> list[dict]:
    return [
        cfg for cfg in configs
        if _class_name(cfg.get("controller", {}).get("class", "")) == controller_name
    ]


def load_result(strategy_name: str, results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / f"{strategy_name}.parquet"
    if not path.exists():
        print(f"Parquet not found: {path.name}  (run backtest first)")
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as exc:
        print(f"Could not read {path.name}: {exc}")
        return None


# plt 
def plot_regime_timeseries(
    df: pd.DataFrame,
    global_regimes: pd.DataFrame,
    estimator_name: str,
    controller_name: str,
    out_path: Path,
) -> None:
    """
    3-panel chart:
      Panel 1 (tall)   – RV20 (actual) + estimator vol_estimate, regime-shaded
      Panel 2 (strip)  – solid regime colour bar
      Panel 3 (medium) – strategy equity curve vs Buy & Hold, regime-shaded

    """

    # ── Merge strategy data 
    merged = df.join(global_regimes, how="inner")
    if merged.empty:
        print(f"No overlapping dates after joining regimes. Skipping.")
        return

    # The common regime series used for ALL shading in this plot
    regime = merged["global_regime"]  

   
    for col in ("vol_estimate", "equity_curve"):
        if col not in merged.columns:
            print(f"Missing column '{col}' in parquet. Skipping {estimator_name}.")
            return

    eq  = merged["equity_curve"].copy()
    rv20 = merged["rv20"].copy()


    plt.style.use("seaborn-v0_8-darkgrid")

    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[5, 0.7, 2.8], hspace=0.07)
    ax_vol   = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax_vol)
    ax_eq    = fig.add_subplot(gs[2], sharex=ax_vol)

    fig.suptitle(
        f"Volatility Regime Time Series  ·  {estimator_name}  |  {controller_name}\n"
        f"Regimes defined on Realised Volatility (RV20) — identical baseline for all estimators",
        fontsize=13, fontweight="bold", y=0.99,
    )

    #  vol lines + regime shading

    _shade_regimes(ax_vol, merged.index, regime, alpha=0.20)

    # RV20 — the ground truth that defined the regimes
    ax_vol.plot(
        merged.index, rv20,
        color="#7f8c8d", linewidth=1.1, alpha=0.75,
        label="Realised Vol (RV20) — regime basis", zorder=2,
    )

    # Estimator forecast
    ax_vol.plot(
        merged.index, merged["vol_estimate"],
        color="#2c3e50", linewidth=1.5, alpha=0.9,
        label=f"{estimator_name} forecast", zorder=3,
    )

    # Target vol reference line
    ax_vol.axhline(
        0.10, color="#3498db", linestyle="--", linewidth=1.2, alpha=0.7,
        label="Target Vol = 10%", zorder=4,
    )

    ax_vol.set_ylabel("Annualised Volatility", fontsize=11)
    ax_vol.set_ylim(bottom=0)
    ax_vol.tick_params(labelbottom=False)

    legend_handles = _regime_legend_handles() + ax_vol.get_legend_handles_labels()[0]
    ax_vol.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.88)
    _annotate_regime_pcts(ax_vol, regime)

  
    # solid regime colour strip

    _draw_regime_strip(ax_strip, merged.index, regime)
    ax_strip.set_yticks([0.5])
    ax_strip.set_yticklabels(["Regime"], fontsize=8)
    ax_strip.tick_params(labelbottom=False, left=False)
    ax_strip.set_ylim(0, 1)


    # equity curve + regime shading
  
    _shade_regimes(ax_eq, merged.index, regime, alpha=0.20)

    ax_eq.plot(
        merged.index, eq,
        color="#2c3e50", linewidth=1.5, zorder=3,
        label="Strategy Equity",
    )

    if "asset_returns" in merged.columns:
        bh = eq.iloc[0] * np.exp(np.cumsum(merged["asset_returns"].fillna(0)))
        ax_eq.plot(
            merged.index, bh,
            color="#7f8c8d", linewidth=1.0, alpha=0.65,
            linestyle="--", zorder=2, label="Buy & Hold",
        )

    ax_eq.set_ylabel("Portfolio Value", fontsize=11)
    ax_eq.set_xlabel("Date", fontsize=11)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.88)
    ax_eq.tick_params(axis="x", rotation=20)

    _add_regime_stats_table(fig, merged, regime)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅  Saved → {out_path.name}")


# ── plotting helpers ─────
def _shade_regimes(
    ax, index: pd.DatetimeIndex, regime: pd.Series, alpha: float = 0.18
) -> None:
    """Background colour fill per contiguous regime block."""
    if len(index) == 0:
        return
    prev_reg    = None
    block_start = index[0]
    for date, reg in zip(index, regime):
        if reg != prev_reg:
            if prev_reg is not None and prev_reg in REGIME_COLORS:
                ax.axvspan(block_start, date,
                           color=REGIME_COLORS[prev_reg], alpha=alpha,
                           linewidth=0, zorder=1)
            prev_reg    = reg
            block_start = date
    if prev_reg is not None and prev_reg in REGIME_COLORS:
        ax.axvspan(block_start, index[-1],
                   color=REGIME_COLORS[prev_reg], alpha=alpha,
                   linewidth=0, zorder=1)


def _draw_regime_strip(
    ax, index: pd.DatetimeIndex, regime: pd.Series
) -> None:
    
    if len(index) == 0:
        return
    prev_reg    = None
    block_start = index[0]
    for date, reg in zip(index, regime):
        if reg != prev_reg:
            if prev_reg is not None and prev_reg in REGIME_COLORS:
                ax.axvspan(block_start, date,
                           color=REGIME_COLORS[prev_reg], alpha=1.0, linewidth=0)
            prev_reg    = reg
            block_start = date
    if prev_reg is not None and prev_reg in REGIME_COLORS:
        ax.axvspan(block_start, index[-1],
                   color=REGIME_COLORS[prev_reg], alpha=1.0, linewidth=0)


def _regime_legend_handles() -> list:
    return [
        mpatches.Patch(
            facecolor=REGIME_COLORS[r], alpha=0.55,
            label=REGIME_LABELS[r], edgecolor="grey", linewidth=0.4,
        )
        for r in REGIME_ORDER
    ]


def _annotate_regime_pcts(ax, regime: pd.Series) -> None:
    """Top-right annotation of time spent in each regime."""
    total = regime.dropna().shape[0]
    if total == 0:
        return
    lines = [
        f"{REGIME_LABELS[r]}: {(regime == r).sum() / total:.1%}"
        for r in REGIME_ORDER
    ]
    ax.text(
        0.995, 0.975, "\n".join(lines),
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.88, edgecolor="grey"),
    )


def _add_regime_stats_table(
    fig: plt.Figure,
    df: pd.DataFrame, 
    regime: pd.Series,
) -> None:
    """Performance summary table split by global regime, placed below the panels."""
    rows = []
    for r in REGIME_ORDER:
        mask = regime == r
        n    = mask.sum()
        if n == 0:
            continue

        rets        = df.loc[mask, "returns"].fillna(0)
        ann_ret     = rets.mean() * 252
        ann_vol     = rets.std() * np.sqrt(252)
        sharpe      = ann_ret / ann_vol if ann_vol > 0 else np.nan
        rv20_mean   = df.loc[mask, "rv20"].mean()
        vol_est_mean = df.loc[mask, "vol_estimate"].mean()

        rows.append([
            REGIME_LABELS[r],
            f"{n:,}",
            f"{rv20_mean:.2%}",       
            f"{vol_est_mean:.2%}",       
            f"{ann_ret:+.2%}",
            f"{ann_vol:.2%}",
            f"{sharpe:.2f}" if np.isfinite(sharpe) else "—",
        ])

    if not rows:
        return

    col_labels = [
        "Regime", "Days",
        "Avg RV20",          # ground truth
        "Avg Vol Forecast",  # estimator
        "Ann. Return", "Ann. Vol", "Sharpe",
    ]

    table_ax = fig.add_axes([0.06, 0.0, 0.90, 0.058])
    table_ax.axis("off")

    tbl = table_ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)

    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col_i == 0 and row_i > 0:
            label_text = rows[row_i - 1][0]
            for r_key, r_label in REGIME_LABELS.items():
                if r_label == label_text:
                    cell.set_facecolor(REGIME_COLORS[r_key])
                    cell.set_text_props(color="white", fontweight="bold")
                    break
        else:
            cell.set_facecolor("#f5f6fa" if row_i % 2 == 0 else "white")


# ── main ──────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot volatility regime time series by controller type. "
                    "Regimes are always defined on actual RV20 — not on estimator forecasts."
    )
    parser.add_argument(
        "--controller", "-c",
        required=True,
        help=(
            "Controller class name to filter on. "
            "Examples: NaiveScaling | RegimeSwitchController | VarianceScaling"
        ),
    )
    args = parser.parse_args()
    controller_name = args.controller.strip()

    strategies_dir = Env.path("strategies")
    results_dir    = Env.path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Fit global regimes 
    global_regimes = fit_global_regimes()


    all_configs = load_yaml_configs(strategies_dir)
    if not all_configs:
        print(" No YAML configs found in", strategies_dir)
        sys.exit(1)

    matched = filter_by_controller(all_configs, controller_name)
    if not matched:
        available = sorted({
            _class_name(cfg.get("controller", {}).get("class", ""))
            for cfg in all_configs
        })
        print(
            f"No configs found for controller '{controller_name}'.\n"
            f"   Available controllers: {available}"
        )
        sys.exit(1)

    print(
        f"\nController  : {controller_name}"
        f"\nConfigs matched : {len(matched)}"
        f"\nOutput dir  : {results_dir}\n"
        f"{'─'*60}"
    )

    # Generate one plot per matched estimator
    generated = 0
    for cfg in matched:
        strategy_name  = cfg.get("name", cfg["_yaml_stem"])
        est_path       = cfg.get("estimator", {}).get("class", "")
        estimator_name = _class_name(est_path) if est_path else cfg["_yaml_stem"]

        print(f"\n {strategy_name}")
        print(f"   Estimator : {estimator_name}")

        df = load_result(strategy_name, results_dir)
        if df is None:
            continue

        out_path = results_dir / f"regime_ts_{estimator_name}_{controller_name}.png"

        plot_regime_timeseries(
            df, global_regimes, estimator_name, controller_name, out_path
        )
        generated += 1

    print(f"\n{'─'*60}")
    print(f"Done. Generated {generated} regime plot(s) in: {results_dir}")


if __name__ == "__main__":
    main()