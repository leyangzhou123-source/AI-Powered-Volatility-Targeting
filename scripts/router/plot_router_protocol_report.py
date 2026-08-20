"""Router protocol report in presentation style (Router vs Buy-and-Hold baseline)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _as_float_series(x: pd.Series) -> pd.Series:
    return pd.Series(x).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _equity_from_returns(r: pd.Series, initial: float = 1.0) -> pd.Series:
    rr = _as_float_series(r)
    return initial * np.exp(rr.cumsum())


def _drawdown(equity: pd.Series) -> pd.Series:
    e = _as_float_series(equity)
    return e / e.cummax() - 1.0


def _safe_mean(x: pd.Series) -> float:
    a = pd.Series(x).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(a.mean()) if len(a) else float("nan")


def _safe_std(x: pd.Series) -> float:
    a = pd.Series(x).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(a.std(ddof=1)) if len(a) >= 2 else float("nan")


def _sharpe(r: pd.Series, ann: float = 252.0) -> float:
    mu = _safe_mean(r)
    sd = _safe_std(r)
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(ann))


def _sortino(r: pd.Series, ann: float = 252.0) -> float:
    rr = _as_float_series(r)
    dn = rr[rr < 0]
    mu = float(rr.mean())
    ds = float(dn.std(ddof=1)) if len(dn) >= 2 else np.nan
    if not np.isfinite(ds) or ds <= 0:
        return float("nan")
    return float((mu / ds) * np.sqrt(ann))


def _max_dd(equity: pd.Series) -> float:
    dd = _drawdown(equity)
    return float(dd.min()) if len(dd) else float("nan")


def _calmar(r: pd.Series, equity: pd.Series, ann: float = 252.0) -> float:
    mu = _safe_mean(r)
    mdd = _max_dd(equity)
    if not np.isfinite(mu) or not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float((mu * ann) / abs(mdd))


def _ulcer_index(equity: pd.Series) -> float:
    dd = _drawdown(equity)
    if len(dd) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(dd.values))))


def _var95(r: pd.Series) -> float:
    rr = _as_float_series(r)
    return float(np.nanpercentile(rr.values, 5)) if len(rr) else float("nan")


def _win_loss_ratio(r: pd.Series) -> float:
    rr = _as_float_series(r)
    wins = int((rr > 0).sum())
    losses = int((rr < 0).sum())
    if losses == 0:
        return float("nan")
    return float(wins / losses)


def _prep_timeseries(input_dir: Path) -> pd.DataFrame:
    ts = _read_csv(input_dir / "router_vs_bh_timeseries.csv")
    if ts.empty:
        return ts

    if "date" in ts.columns:
        ts["date"] = pd.to_datetime(ts["date"])
        ts = ts.sort_values("date")

    if "router_return" not in ts.columns or "bh_return" not in ts.columns:
        return pd.DataFrame()

    if "router_equity" not in ts.columns:
        ts["router_equity"] = _equity_from_returns(ts["router_return"])
    if "bh_equity" not in ts.columns:
        ts["bh_equity"] = _equity_from_returns(ts["bh_return"])

    ts["router_dd"] = _drawdown(ts["router_equity"])
    ts["bh_dd"] = _drawdown(ts["bh_equity"])

    ts["router_roll_vol"] = _as_float_series(ts["router_return"]).rolling(21).std() * np.sqrt(252)
    ts["bh_roll_vol"] = _as_float_series(ts["bh_return"]).rolling(21).std() * np.sqrt(252)

    return ts


def plot_dim1_risk_adjusted(ts: pd.DataFrame, out_dir: Path):
    if ts.empty:
        return
    x = ts["date"] if "date" in ts.columns else ts.index

    rr = _as_float_series(ts["router_return"])
    br = _as_float_series(ts["bh_return"])
    re = _as_float_series(ts["router_equity"])
    be = _as_float_series(ts["bh_equity"])

    rows = [
        [
            "Router",
            f"{_sharpe(rr):.2f}",
            f"{_sortino(rr):.2f}",
            f"{_calmar(rr, re):.2f}",
            f"{(re.iloc[-1] / re.iloc[0] - 1):.2%}",
        ],
        [
            "Buy&Hold",
            f"{_sharpe(br):.2f}",
            f"{_sortino(br):.2f}",
            f"{_calmar(br, be):.2f}",
            f"{(be.iloc[-1] / be.iloc[0] - 1):.2%}",
        ],
    ]

    fig, (ax, ax_tbl) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3.3, 1.1]},
        constrained_layout=True,
    )
    ax.plot(x, re, label="Router", linewidth=1.7)
    ax.plot(x, be, label="Buy&Hold", linewidth=1.7)
    ax.set_title("Dimension 1: Risk-Adjusted Returns (Router vs Buy&Hold)", fontsize=16)
    ax.set_ylabel("Equity")
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Strategy", "Sharpe", "Sortino", "Calmar", "Total Return"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    _save(fig, out_dir / "1_Risk_Adjusted_Returns.png")


def plot_dim2_vol_control(ts: pd.DataFrame, out_dir: Path, target_vol: float = 0.10):
    if ts.empty:
        return
    x = ts["date"] if "date" in ts.columns else ts.index

    rv = ts["router_roll_vol"]
    bv = ts["bh_roll_vol"]

    rv_mean = _safe_mean(rv)
    bv_mean = _safe_mean(bv)
    rv_cv = _safe_std(rv) / rv_mean if np.isfinite(rv_mean) and rv_mean > 0 else np.nan
    bv_cv = _safe_std(bv) / bv_mean if np.isfinite(bv_mean) and bv_mean > 0 else np.nan

    rows = [
        ["Router", f"{rv_mean:.2%}", f"{rv_cv:.3f}", f"{(rv_mean - target_vol):.2%}"],
        ["Buy&Hold", f"{bv_mean:.2%}", f"{bv_cv:.3f}", f"{(bv_mean - target_vol):.2%}"],
    ]

    fig, (ax, ax_tbl) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3.3, 1.1]},
        constrained_layout=True,
    )
    ax.plot(x, rv, label="Router Rolling Vol", linewidth=1.5)
    ax.plot(x, bv, label="Buy&Hold Rolling Vol", linewidth=1.5)
    ax.axhline(target_vol, color="black", linestyle="--", alpha=0.6, label="Target Vol")
    ax.set_title("Dimension 2: Volatility Control Ability", fontsize=16)
    ax.set_ylabel("Annualized Vol")
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Strategy", "Avg Realized Vol", "Vol CV", "Target Deviation"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    _save(fig, out_dir / "2_Vol_Control.png")


def plot_dim3_extreme_risk(ts: pd.DataFrame, out_dir: Path):
    if ts.empty:
        return
    x = ts["date"] if "date" in ts.columns else ts.index

    rr = _as_float_series(ts["router_return"])
    br = _as_float_series(ts["bh_return"])
    re = _as_float_series(ts["router_equity"])
    be = _as_float_series(ts["bh_equity"])

    rdd = _drawdown(re)
    bdd = _drawdown(be)

    rows = [
        [
            "Router",
            f"{_max_dd(re):.2%}",
            f"{_ulcer_index(re):.4f}",
            f"{_var95(rr):.2%}",
            f"{int((rdd < -0.05).sum())} Days",
        ],
        [
            "Buy&Hold",
            f"{_max_dd(be):.2%}",
            f"{_ulcer_index(be):.4f}",
            f"{_var95(br):.2%}",
            f"{int((bdd < -0.05).sum())} Days",
        ],
    ]

    fig, (ax, ax_tbl) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3.3, 1.1]},
        constrained_layout=True,
    )
    ax.fill_between(x, rdd, 0, alpha=0.35, label="Router DD")
    ax.fill_between(x, bdd, 0, alpha=0.35, label="Buy&Hold DD")
    ax.set_title("Dimension 3: Extreme Risk Metrics", fontsize=16)
    ax.set_ylabel("Drawdown")
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Strategy", "Max Drawdown", "Ulcer Index", "VaR (95%)", "Days with DD > 5%"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    _save(fig, out_dir / "3_Extreme_Risk.png")


def plot_dim4_trading_efficiency(ts: pd.DataFrame, router_oos: pd.DataFrame, out_dir: Path):
    if ts.empty:
        return

    rr = _as_float_series(ts["router_return"])
    br = _as_float_series(ts["bh_return"])

    switch_rate = np.nan
    switch_count = np.nan
    if not router_oos.empty:
        if "switch_rate" in router_oos.columns:
            switch_rate = float(pd.to_numeric(router_oos["switch_rate"], errors="coerce").mean())
        if "switch_count" in router_oos.columns:
            switch_count = float(pd.to_numeric(router_oos["switch_count"], errors="coerce").sum())

    rows = [
        ["Router", f"{switch_rate:.3%}" if np.isfinite(switch_rate) else "nan", f"{switch_count:.0f}" if np.isfinite(switch_count) else "nan", f"{_win_loss_ratio(rr):.2f}"],
        ["Buy&Hold", "0.000%", "0", f"{_win_loss_ratio(br):.2f}"],
    ]

    fig, (ax, ax_tbl) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3.3, 1.1]},
        constrained_layout=True,
    )

    # show split-level switching as bars if available
    if not router_oos.empty and "split" in router_oos.columns and "switch_rate" in router_oos.columns:
        x = pd.to_numeric(router_oos["split"], errors="coerce")
        y = pd.to_numeric(router_oos["switch_rate"], errors="coerce")
        ax.bar(x, y, alpha=0.75)
        ax.set_xlabel("Split")
        ax.set_ylabel("Switch Rate")
        ax.set_title("Dimension 4: Trading Efficiency (Router Switching Cost)", fontsize=16)
    else:
        ax.axis("off")
        ax.set_title("Dimension 4: Trading Efficiency", fontsize=16)

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Strategy", "Avg Switch Rate", "Total Switch Count", "Win/Loss Ratio"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    _save(fig, out_dir / "4_Trading_Costs.png")


def plot_meta_analysis(summary: pd.DataFrame, out_dir: Path):
    if summary.empty:
        return
    row = summary.iloc[0]

    router_sharpe = float(row.get("overall_sharpe", np.nan))
    bh_sharpe = float(row.get("bh_overall_sharpe", np.nan))
    router_cvar = float(row.get("overall_cvar_95", np.nan))
    bh_cvar = float(row.get("bh_overall_cvar_95", np.nan))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(router_cvar, router_sharpe, s=220, label="Router")
    ax.scatter(bh_cvar, bh_sharpe, s=220, label="Buy&Hold")
    ax.set_xlabel("CVaR 95% (lower better)")
    ax.set_ylabel("Sharpe (higher better)")
    ax.set_title("Meta-Analysis: Tail Risk vs Risk-Adjusted Return")
    ax.legend()
    _save(fig, out_dir / "5_Meta_Analysis.png")


def write_comparison_table(summary: pd.DataFrame, router_oos: pd.DataFrame, out_dir: Path):
    if summary.empty:
        return
    row = summary.iloc[0]
    out = pd.DataFrame(
        [
            {
                "strategy": "Router",
                "overall_sharpe": row.get("overall_sharpe"),
                "overall_calmar": row.get("overall_calmar"),
                "overall_cvar_95": row.get("overall_cvar_95"),
            },
            {
                "strategy": "Buy&Hold",
                "overall_sharpe": row.get("bh_overall_sharpe"),
                "overall_calmar": row.get("bh_overall_calmar"),
                "overall_cvar_95": row.get("bh_overall_cvar_95"),
            },
        ]
    )
    if not router_oos.empty and "excess_sharpe" in router_oos.columns:
        out["avg_split_excess_sharpe"] = float(pd.to_numeric(router_oos["excess_sharpe"], errors="coerce").mean())
    out.to_csv(out_dir / "router_vs_bh_comparison.csv", index=False)

    disp = out.copy()
    for c in disp.columns:
        if c == "strategy":
            continue
        disp[c] = pd.to_numeric(disp[c], errors="coerce").map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")
    ax.set_title("Router vs Buy&Hold Comparison Table", fontsize=14, pad=10)
    table = ax.table(cellText=disp.values, colLabels=disp.columns.tolist(), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    _save(fig, out_dir / "table.png")


def main():
    parser = argparse.ArgumentParser(description="Plot presentation-style Router vs Buy&Hold report.")
    parser.add_argument("--input-dir", default="results/evaluation/router_protocol")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (input_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_csv(input_dir / "summary.csv")
    router_oos = _read_csv(input_dir / "router_oos_diagnostics.csv")
    ts = _prep_timeseries(input_dir)

    plot_dim1_risk_adjusted(ts, out_dir)
    plot_dim2_vol_control(ts, out_dir)
    plot_dim3_extreme_risk(ts, out_dir)
    plot_dim4_trading_efficiency(ts, router_oos, out_dir)
    plot_meta_analysis(summary, out_dir)
    write_comparison_table(summary, router_oos, out_dir)

    print("=" * 60)
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {out_dir}")
    print("Saved presentation-style Router vs Buy&Hold report:")
    print("  1_Risk_Adjusted_Returns.png")
    print("  2_Vol_Control.png")
    print("  3_Extreme_Risk.png")
    print("  4_Trading_Costs.png")
    print("  5_Meta_Analysis.png")
    print("  table.png")
    print("  router_vs_bh_comparison.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
