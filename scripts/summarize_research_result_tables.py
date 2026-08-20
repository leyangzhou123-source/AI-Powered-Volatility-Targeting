from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIME_SERIES_FILE_CANDIDATES = (
    "router_vs_bh_timeseries.csv",
    "result.csv",
    "equity_curve.csv",
)

RETURN_COLUMN_CANDIDATES = (
    "returns_with_rf",
    "raw_returns_with_rf",
    "router_return",
    "strategy_returns",
    "strategy_return",
    "return",
    "returns",
)

EQUITY_COLUMN_CANDIDATES = (
    "equity_curve_with_rf",
    "router_equity",
    "equity_curve",
    "strategy_equity",
    "equity",
)

PAIR_COLUMN_CANDIDATES = (
    "active_pair",
    "selected_pair",
    "selected_pair_name",
)

DATE_COLUMN_CANDIDATES = (
    "date",
    "datetime",
    "timestamp",
)


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    by_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _pick_time_series_csv(path: Path) -> Path:
    if path.is_file():
        return path

    for name in TIME_SERIES_FILE_CANDIDATES:
        candidate = path / name
        if candidate.exists():
            return candidate

    csvs = sorted(p for p in path.glob("*.csv") if _is_usable_time_series_csv(p))
    scored: list[tuple[int, Path]] = []
    for csv_path in csvs:
        try:
            cols = pd.read_csv(csv_path, nrows=0).columns
        except Exception:
            continue
        lower_cols = {col.lower() for col in cols}
        score = 0
        if lower_cols.intersection(RETURN_COLUMN_CANDIDATES):
            score += 5
        if lower_cols.intersection(EQUITY_COLUMN_CANDIDATES):
            score += 4
        if lower_cols.intersection(PAIR_COLUMN_CANDIDATES):
            score += 2
        if lower_cols.intersection(DATE_COLUMN_CANDIDATES):
            score += 1
        if score:
            scored.append((score, csv_path))

    if not scored:
        raise FileNotFoundError(f"No usable strategy time-series CSV found in {path}")

    return max(scored, key=lambda item: (item[0], -len(item[1].name)))[1]


def _is_usable_time_series_csv(path: Path) -> bool:
    name = path.name.lower()
    if "switch_log" in name or name.endswith(".summary.csv") or name == "summary.csv":
        return False
    try:
        cols = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    lower_cols = {col.lower() for col in cols}
    has_return = bool(lower_cols.intersection(RETURN_COLUMN_CANDIDATES))
    has_equity = bool(lower_cols.intersection(EQUITY_COLUMN_CANDIDATES))
    return has_return or has_equity


def _expand_group_path(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if _is_usable_time_series_csv(path) else []

    child_dirs = sorted(p for p in path.iterdir() if p.is_dir())
    direct_csvs = sorted(p for p in path.glob("*.csv") if _is_usable_time_series_csv(p))

    expanded: list[Path] = []
    for child in child_dirs:
        try:
            _pick_time_series_csv(child)
        except FileNotFoundError:
            continue
        expanded.append(child)
    expanded.extend(direct_csvs)

    if not child_dirs and len(direct_csvs) == 1 and direct_csvs[0].name in TIME_SERIES_FILE_CANDIDATES:
        return [path]

    if not expanded:
        try:
            _pick_time_series_csv(path)
        except FileNotFoundError:
            return []
        return [path]

    return expanded


def _ann_factor_from_folder(path: Path, dates: pd.Series | None) -> float:
    folder = path if path.is_dir() else path.parent
    summary_path = folder / "summary.csv"
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            if "ann_factor" in summary.columns and len(summary):
                ann = float(summary["ann_factor"].iloc[0])
                if math.isfinite(ann) and ann > 0:
                    return ann
        except Exception:
            pass

    if dates is not None:
        parsed = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
        if len(parsed) >= 2:
            span_days = max((parsed.iloc[-1] - parsed.iloc[0]).days, 1)
            obs_per_calendar_year = len(parsed) / span_days * 365.25
            if obs_per_calendar_year > 300:
                return 365.0

    return 252.0


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    return np.exp(returns.cumsum())


def _max_drawdown(equity: pd.Series) -> float:
    e = _clean_numeric(equity)
    if len(e) < 2:
        return float("nan")
    drawdown = e / e.cummax() - 1.0
    return float(-drawdown.min())


def _cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    r = _clean_numeric(returns)
    if len(r) < 10:
        return float("nan")
    cutoff = float(np.quantile(r, alpha))
    tail = r[r <= cutoff]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def _sharpe(returns: pd.Series, ann_factor: float) -> float:
    r = _clean_numeric(returns)
    if len(r) < 2:
        return float("nan")
    std = float(r.std(ddof=1))
    if std <= 0 or not math.isfinite(std):
        return float("nan")
    return float((r.mean() / std) * math.sqrt(ann_factor))


def _sortino(returns: pd.Series, ann_factor: float) -> float:
    r = _clean_numeric(returns)
    if len(r) < 2:
        return float("nan")
    downside = r[r < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) >= 2 else float("nan")
    if downside_std <= 0 or not math.isfinite(downside_std):
        return float("nan")
    return float((r.mean() / downside_std) * math.sqrt(ann_factor))


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _switch_count(df: pd.DataFrame, pair_col: str | None) -> int:
    if "switch_executed" in df.columns:
        return int(_as_bool_series(df["switch_executed"]).sum())

    if pair_col is None:
        return 0
    selected = df[pair_col].dropna()
    if selected.empty:
        return 0
    return int(selected.ne(selected.shift()).sum() - 1)


def summarize_csv(
    path: Path,
    strategy_name: str | None = None,
    return_col: str | None = None,
    equity_col: str | None = None,
    pair_col: str | None = None,
    date_col: str | None = None,
    ann_factor: float | None = None,
    cvar_alpha: float = 0.05,
) -> dict[str, float | int | str]:
    csv_path = _pick_time_series_csv(path)
    df = pd.read_csv(csv_path)

    return_col = return_col or _first_existing_column(df, RETURN_COLUMN_CANDIDATES)
    if return_col is None:
        raise ValueError(f"No return column found in {csv_path}")

    equity_col = equity_col or _first_existing_column(df, EQUITY_COLUMN_CANDIDATES)
    pair_col = pair_col or _first_existing_column(df, PAIR_COLUMN_CANDIDATES)
    date_col = date_col or _first_existing_column(df, DATE_COLUMN_CANDIDATES)

    returns = _clean_numeric(df[return_col])
    if returns.empty:
        raise ValueError(f"Return column {return_col!r} has no numeric values in {csv_path}")

    equity = _clean_numeric(df[equity_col]) if equity_col is not None else _equity_from_returns(returns)
    ann = ann_factor or _ann_factor_from_folder(path, df[date_col] if date_col else None)
    ann_return = float(returns.mean() * ann)
    ann_vol = float(returns.std(ddof=1) * math.sqrt(ann)) if len(returns) >= 2 else float("nan")
    max_dd = _max_drawdown(equity)

    folder_name = path.name if path.is_dir() else path.stem
    return {
        "strategy_name": strategy_name or folder_name,
        "ann_vol": ann_vol,
        "ann_return": ann_return,
        "sharpe": _sharpe(returns, ann),
        "max_dd": max_dd,
        "cvar": _cvar(returns, cvar_alpha),
        "sortino": _sortino(returns, ann),
        "calmar": ann_return / max_dd if max_dd and math.isfinite(max_dd) else float("nan"),
        "switch_count": _switch_count(df, pair_col),
        "independent_pair_use": int(df[pair_col].dropna().nunique()) if pair_col else 0,
        "source_csv": str(csv_path),
    }


def _write_folder_table(row: dict[str, float | int | str], folder: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = str(row["strategy_name"]).replace(" ", "_").replace("/", "_")
    out_path = output_dir / f"{name}_strategy_metrics.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize strategy time-series CSVs into research-result metric tables."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Folders or CSV files to summarize.")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand each input folder into direct child result folders/CSVs.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Write one table per input here.")
    parser.add_argument("--combined-output", type=Path, default=None, help="Optional combined CSV path.")
    parser.add_argument("--strategy-name", default=None, help="Override strategy name for one input path.")
    parser.add_argument("--return-col", default=None, help="Override return column.")
    parser.add_argument("--equity-col", default=None, help="Override equity column.")
    parser.add_argument("--pair-col", default=None, help="Override selected-pair column.")
    parser.add_argument("--date-col", default=None, help="Override date column.")
    parser.add_argument("--ann-factor", type=float, default=None, help="Override annualization factor.")
    parser.add_argument("--cvar-alpha", type=float, default=0.05, help="Tail probability for CVaR.")
    args = parser.parse_args()

    paths = [item for path in args.paths for item in (_expand_group_path(path) if args.expand else [path])]
    if args.strategy_name and len(paths) != 1:
        raise ValueError("--strategy-name can only be used with one input path.")

    rows = [
        summarize_csv(
            path,
            strategy_name=args.strategy_name,
            return_col=args.return_col,
            equity_col=args.equity_col,
            pair_col=args.pair_col,
            date_col=args.date_col,
            ann_factor=args.ann_factor,
            cvar_alpha=args.cvar_alpha,
        )
        for path in paths
    ]

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    if args.output_dir:
        for row, path in zip(rows, paths, strict=True):
            folder = path if path.is_dir() else path.parent
            out_path = _write_folder_table(row, folder, args.output_dir)
            print(f"Wrote {out_path}")

    if args.combined_output:
        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.combined_output, index=False)
        print(f"Wrote {args.combined_output}")


if __name__ == "__main__":
    main()
