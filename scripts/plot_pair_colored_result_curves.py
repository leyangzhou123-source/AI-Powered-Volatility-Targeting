from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


DATE_COLUMNS = ("date", "datetime", "timestamp")
PAIR_COLUMNS = ("active_pair", "selected_pair", "selected_pair_name")
RETURN_COLUMNS = ("returns_with_rf", "router_return", "strategy_return", "strategy_returns", "returns")
EQUITY_COLUMNS = ("equity_curve_with_rf", "router_equity", "strategy_equity", "equity_curve", "equity")
BENCHMARK_RETURN_COLUMNS = ("bh_return", "buy_hold_return", "raw_returns_with_rf")
BENCHMARK_EQUITY_COLUMNS = ("bh_equity", "buy_hold_equity")
VOL_COLUMNS = (
    "active_realized_vol",
    "realized_vol",
    "volatility",
    "vol_estimate",
    "selected_realized_vol",
)

PAIR_COLORS = (
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#7f7f7f",
    "#003f5c",
    "#58508d",
    "#bc5090",
    "#ff6361",
    "#ffa600",
    "#006d2c",
    "#08519c",
    "#a50f15",
    "#54278f",
    "#8c6d31",
)


def _first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    columns = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    return None


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _rolling_realized_vol(returns: pd.Series, ann_factor: float, window: int) -> pd.Series:
    r = _as_float(returns)
    return r.rolling(window, min_periods=max(2, window // 3)).std(ddof=1) * np.sqrt(ann_factor)


def _ann_factor(dates: pd.Series) -> float:
    parsed = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(parsed) < 2:
        return 252.0
    span_days = max((parsed.iloc[-1] - parsed.iloc[0]).days, 1)
    obs_per_year = len(parsed) / span_days * 365.25
    return 365.0 if obs_per_year > 300 else 252.0


def _segments_by_pair(
    dates: pd.Series,
    values: pd.Series,
    pairs: pd.Series,
    color_map: dict[str, tuple[float, float, float, float]],
) -> list[LineCollection]:
    x = mdates.date2num(pd.to_datetime(dates))
    y = _as_float(values).to_numpy(dtype=float)
    pair_values = pairs.astype(str).to_numpy()

    collections: list[LineCollection] = []
    for pair, color in color_map.items():
        segments = []
        for idx in range(len(x) - 1):
            if pair_values[idx] != pair or pair_values[idx + 1] != pair:
                continue
            if not np.isfinite(y[idx]) or not np.isfinite(y[idx + 1]):
                continue
            segments.append([(x[idx], y[idx]), (x[idx + 1], y[idx + 1])])
        if segments:
            collections.append(LineCollection(segments, colors=[color], linewidths=2.8, zorder=4))
    return collections


def _shade_pair_runs(ax: plt.Axes, dates: pd.Series, pairs: pd.Series, color_map: dict[str, tuple]) -> None:
    parsed_dates = pd.to_datetime(dates).reset_index(drop=True)
    pair_values = pairs.astype(str).reset_index(drop=True)
    start = 0
    for idx in range(1, len(pair_values) + 1):
        if idx < len(pair_values) and pair_values.iloc[idx] == pair_values.iloc[start]:
            continue
        pair = pair_values.iloc[start]
        color = color_map[pair]
        left = parsed_dates.iloc[start]
        right = parsed_dates.iloc[idx - 1]
        if idx < len(parsed_dates):
            right = parsed_dates.iloc[idx]
        ax.axvspan(left, right, color=color, alpha=0.18, linewidth=0, zorder=0)
        start = idx


def _benchmark_equity_from_returns(returns: pd.Series, strategy_equity: pd.Series) -> pd.Series:
    clean_returns = _as_float(returns).fillna(0.0)
    first_equity = _as_float(strategy_equity).dropna()
    start_value = float(first_equity.iloc[0]) if len(first_equity) else 1.0
    return start_value * (1.0 + clean_returns).cumprod()


def _load_external_benchmark(
    benchmark_path: Path,
    target_dates: pd.Series,
    strategy_equity: pd.Series,
) -> tuple[pd.Series | None, pd.Series | None, str]:
    bench = pd.read_csv(benchmark_path)
    bench_date_col = _first_column(bench, DATE_COLUMNS)
    bench_return_col = _first_column(bench, RETURN_COLUMNS + BENCHMARK_RETURN_COLUMNS)
    bench_equity_col = _first_column(bench, EQUITY_COLUMNS + BENCHMARK_EQUITY_COLUMNS)
    if bench_date_col is None or bench_return_col is None:
        raise ValueError(f"{benchmark_path} needs date and return columns.")

    bench = bench.copy()
    bench["_plot_date"] = pd.to_datetime(bench[bench_date_col], errors="coerce")
    bench = bench.dropna(subset=["_plot_date"]).sort_values("_plot_date")

    target_index = pd.Index(pd.to_datetime(target_dates), name="_plot_date")
    bench = bench.set_index("_plot_date")
    returns = _as_float(bench[bench_return_col]).reindex(target_index).reset_index(drop=True)
    if bench_equity_col is not None:
        equity = _as_float(bench[bench_equity_col]).reindex(target_index).reset_index(drop=True)
    else:
        equity = _benchmark_equity_from_returns(returns, strategy_equity)

    label = "Benchmark"
    if "benchmark_name" in bench.columns:
        names = bench["benchmark_name"].dropna().astype(str).unique()
        if len(names):
            label = names[0]

    return returns, equity, label


def plot_result_curves(
    csv_path: Path,
    output_path: Path,
    title: str,
    benchmark_path: Path | None = None,
    benchmark_label: str | None = None,
    rolling_window: int = 21,
) -> None:
    df = pd.read_csv(csv_path)
    date_col = _first_column(df, DATE_COLUMNS)
    pair_col = _first_column(df, PAIR_COLUMNS)
    return_col = _first_column(df, RETURN_COLUMNS)
    equity_col = _first_column(df, EQUITY_COLUMNS)
    benchmark_return_col = _first_column(df, BENCHMARK_RETURN_COLUMNS)
    benchmark_equity_col = _first_column(df, BENCHMARK_EQUITY_COLUMNS)
    vol_col = _first_column(df, VOL_COLUMNS)

    missing = [
        name
        for name, col in {
            "date": date_col,
            "pair": pair_col,
            "return": return_col,
            "equity": equity_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    df = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    pairs = df[pair_col].astype(str).fillna("missing")
    equity = _as_float(df[equity_col])
    if benchmark_path is not None:
        benchmark_returns, benchmark_equity, detected_benchmark_label = _load_external_benchmark(
            benchmark_path,
            dates,
            equity,
        )
        benchmark_legend_label = benchmark_label or detected_benchmark_label
    else:
        benchmark_returns = _as_float(df[benchmark_return_col]) if benchmark_return_col else None
        if benchmark_equity_col is not None:
            benchmark_equity = _as_float(df[benchmark_equity_col])
        elif benchmark_returns is not None:
            benchmark_equity = _benchmark_equity_from_returns(benchmark_returns, equity)
        else:
            benchmark_equity = None
        benchmark_legend_label = benchmark_label or "Buy & hold / raw benchmark"

    if vol_col is None:
        vol = _rolling_realized_vol(df[return_col], _ann_factor(df[date_col]), rolling_window)
        vol_label = f"{rolling_window}D realized vol"
    else:
        vol = _as_float(df[vol_col])
        vol_label = vol_col
    benchmark_vol = (
        _rolling_realized_vol(benchmark_returns, _ann_factor(df[date_col]), rolling_window)
        if benchmark_returns is not None
        else None
    )

    valid = dates.notna() & pairs.notna() & equity.notna()
    df = df.loc[valid].reset_index(drop=True)
    dates = dates.loc[valid].reset_index(drop=True)
    pairs = pairs.loc[valid].reset_index(drop=True)
    equity = equity.loc[valid].reset_index(drop=True)
    vol = vol.loc[valid].reset_index(drop=True)
    if benchmark_equity is not None:
        benchmark_equity = benchmark_equity.loc[valid].reset_index(drop=True)
    if benchmark_vol is not None:
        benchmark_vol = benchmark_vol.loc[valid].reset_index(drop=True)

    unique_pairs = list(dict.fromkeys(pairs.tolist()))
    color_map = {pair: PAIR_COLORS[i % len(PAIR_COLORS)] for i, pair in enumerate(unique_pairs)}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.08},
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    panel_specs = (
        (axes[0], vol, vol_label, benchmark_vol),
        (axes[1], equity, equity_col, benchmark_equity),
    )
    for ax, values, label, benchmark_values in panel_specs:
        _shade_pair_runs(ax, dates, pairs, color_map)
        if benchmark_values is not None:
            ax.plot(
                dates,
                benchmark_values,
                color="#111827",
                linewidth=1.8,
                linestyle=(0, (1.0, 2.0)),
                alpha=0.9,
                zorder=2,
            )
        for collection in _segments_by_pair(dates, values, pairs, color_map):
            ax.add_collection(collection)
        ax.plot(dates, values, color="#111827", linewidth=0.7, alpha=0.18, zorder=3)
        ax.autoscale()
        ax.set_ylabel(label)
        ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))

    legend_handles = [
        Line2D([0], [0], color=color_map[pair], lw=3, label=pair) for pair in unique_pairs
    ]
    if benchmark_returns is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#374151",
                lw=2.0,
                linestyle=(0, (1.0, 2.0)),
                label=benchmark_legend_label,
            )
        )
    axes[1].legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.72),
        ncol=1,
        frameon=False,
        fontsize=8.2,
        title="Strategy pair",
        title_fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.68, top=0.92, bottom=0.10)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot volatility and equity curves colored by selected strategy pair."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--benchmark-label", default=None)
    parser.add_argument("--rolling-window", type=int, default=21)
    args = parser.parse_args()

    plot_result_curves(
        csv_path=args.csv_path,
        output_path=args.output,
        title=args.title,
        benchmark_path=args.benchmark,
        benchmark_label=args.benchmark_label,
        rolling_window=args.rolling_window,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
