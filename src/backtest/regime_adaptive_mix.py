import numpy as np
import pandas as pd

from pathlib import Path

from src.backtest.base import Engine
from src.env import Env

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None


REGIME_ORDER = ["Low", "Mid", "High"]
OBJECTIVE_ORDER = ["Sharpe", "MaxDrawdown", "CVaR", "Ultimate"]


def _simple_to_log(simple_returns: pd.Series) -> pd.Series:
    s = pd.Series(simple_returns).astype(float)
    return np.log1p(s.clip(lower=-0.999999999))


def _annualized_vol(returns: pd.Series, ann_factor: float = 252.0) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def _annualized_return(returns: pd.Series, ann_factor: float = 252.0) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if r.empty:
        return float("nan")
    wealth = float((1.0 + r).prod())
    if wealth <= 0:
        return float("nan")
    return float(wealth ** (ann_factor / len(r)) - 1.0)


def _sharpe_ratio(returns: pd.Series, ann_factor: float = 252.0) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    sigma = float(r.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(np.sqrt(ann_factor) * r.mean() / sigma)


def _max_drawdown_loss(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(-dd.min())


def _cvar_loss(returns: pd.Series, alpha: float = 0.95) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if r.empty:
        return float("nan")
    cutoff = float(np.percentile(r, (1.0 - alpha) * 100.0))
    tail = r[r <= cutoff]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def _returns_from_result(df: pd.DataFrame) -> pd.Series:
    for col in ["returns_with_rf", "strategy_returns", "returns", "returns_no_rf"]:
        if col in df.columns:
            log_ret = pd.Series(df[col], index=df.index).astype(float)
            return pd.Series(np.expm1(log_ret), index=df.index, name=col)
    return pd.Series(index=df.index, dtype=float)


def _vectorized_sharpe(frame: pd.DataFrame, ann_factor: float) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    mu = frame.mean(axis=0, skipna=True)
    sigma = frame.std(axis=0, ddof=1, skipna=True)
    out = np.sqrt(ann_factor) * mu / sigma
    out = out.where(np.isfinite(out), np.nan)
    return out.astype(float)


def _vectorized_max_drawdown_loss(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    wealth = (1.0 + frame.fillna(0.0)).cumprod(axis=0)
    peak = wealth.cummax(axis=0)
    dd = wealth / peak - 1.0
    return (-dd.min(axis=0, skipna=True)).astype(float)


def _vectorized_cvar_loss(frame: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    arr = frame.to_numpy(dtype=float)
    out = np.full(arr.shape[1], np.nan, dtype=float)

    for j in range(arr.shape[1]):
        col = arr[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        cutoff = float(np.percentile(col, (1.0 - alpha) * 100.0))
        tail = col[col <= cutoff]
        if tail.size == 0:
            continue
        out[j] = float(-tail.mean())

    return pd.Series(out, index=frame.columns, dtype=float)


def _vectorized_mean_turnover(weight_frame: pd.DataFrame) -> pd.Series:
    if weight_frame.empty:
        return pd.Series(dtype=float)
    return weight_frame.diff().abs().fillna(0.0).mean(axis=0, skipna=True).astype(float)


class RollingRegimeMixEngine(Engine):
    def __init__(self, name: str = "rolling_regime_mix", config: dict | None = None):
        self.name = name
        self.config = config or {}
        self.result: pd.DataFrame | None = None
        self.summary_table: pd.DataFrame | None = None
        self.initial_winners: pd.DataFrame | None = None
        self.selection_history: pd.DataFrame | None = None
        self.weight_panel: pd.DataFrame | None = None
        self._hmm_transition_cache: dict[tuple[str, int], np.ndarray] = {}

    def _results_dir(self) -> Path:
        path = self.config.get("results_dir")
        if path:
            return Path(path)
        return Env.path("results")

    def _load_strategy_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        results_dir = self._results_dir()
        strategy_filter = self.config.get("strategy_filter")
        exclude_estimators = {
            str(x).strip().lower()
            for x in self.config.get("exclude_estimators", [])
            if str(x).strip()
        }
        exclude_controllers = {
            str(x).strip().lower()
            for x in self.config.get("exclude_controllers", [])
            if str(x).strip()
        }
        return_frames: list[pd.Series] = []
        weight_frames: list[pd.Series] = []

        for path in sorted(results_dir.glob("*.parquet")):
            if "__" not in path.stem:
                continue
            if strategy_filter and path.stem not in strategy_filter:
                continue

            estimator_name, controller_name = path.stem.split("__", 1)
            if estimator_name.strip().lower() in exclude_estimators:
                continue
            if controller_name.strip().lower() in exclude_controllers:
                continue

            try:
                df = pd.read_parquet(path)
                df.index = pd.to_datetime(df.index)
                if getattr(df.index, "tz", None) is not None:
                    df.index = df.index.tz_localize(None)
                df.index = df.index.normalize()
                df = df.sort_index()

                returns = _returns_from_result(df).dropna()
                if returns.empty:
                    continue
                returns.name = path.stem
                return_frames.append(returns)

                if "weight" in df.columns:
                    weights = pd.Series(df["weight"], index=df.index, dtype=float)
                else:
                    weights = pd.Series(np.nan, index=df.index, dtype=float)
                weights.name = path.stem
                weight_frames.append(weights)
            except Exception as exc:
                print(f"Skip {path.name}: {exc}")

        if not return_frames:
            raise ValueError(
                "No estimator/controller result parquets found. "
                "Run scripts/run_all_combinations.py first."
            )

        return_panel = pd.concat(return_frames, axis=1).sort_index()
        return_panel = return_panel.loc[:, ~return_panel.columns.duplicated()].copy()

        weight_panel = pd.concat(weight_frames, axis=1).sort_index()
        weight_panel = weight_panel.loc[:, ~weight_panel.columns.duplicated()].copy()
        weight_panel = weight_panel.reindex(return_panel.index)

        return return_panel, weight_panel

    def _load_intraday_proxy(self) -> pd.Series:
        parquet_name = self.config.get(
            "proxy_parquet_name", "SP500_Intraday_RealizedVol.parquet"
        )
        vol_col = self.config.get("proxy_vol_col", "realized_vol")
        min_coverage_ratio = float(self.config.get("min_coverage_ratio", 0.95))

        path = Env.path("processed") / parquet_name
        if not path.exists():
            raise FileNotFoundError(f"Missing intraday realized vol parquet: {path}")

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df.sort_index()

        if vol_col not in df.columns:
            raise ValueError(f"Missing proxy column '{vol_col}' in {path}")

        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        if "coverage" in df.columns:
            df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
            df = df[df["coverage"] >= min_coverage_ratio].copy()

        out = df[vol_col].astype(float).copy()
        out.name = "intraday_vol_proxy"
        return out

    def _load_daily_returns_signal(self) -> pd.Series:
        data_path = self.config.get("daily_returns_path")
        if data_path:
            path = Path(data_path)
        else:
            path = Env.path("processed") / "ES_Daily_Processed.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Missing daily returns parquet: {path}")

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df.sort_index()

        preferred_cols = ["returns_clean", "returns", "log_return", "daily_return"]
        ret_col = next((col for col in preferred_cols if col in df.columns), None)
        if ret_col is None:
            raise ValueError(
                f"Could not find daily returns column in {path}. "
                f"Tried {preferred_cols}; found {list(df.columns)}"
            )

        out = pd.to_numeric(df[ret_col], errors="coerce").astype(float)
        out.name = "daily_returns_signal"
        return out

    def _classify_regime(self, value: float, q_low: float, q_high: float) -> str | None:
        if not np.isfinite(value):
            return None
        if value <= q_low:
            return "Low"
        if value <= q_high:
            return "Mid"
        return "High"

    def _fit_hmm_regimes(
        self,
        signal: pd.Series,
        input_kind: str = "intraday_vol",
        as_of_date: pd.Timestamp | None = None,
    ) -> tuple[pd.Series, GaussianHMM | None, dict[int, str]]:
        if GaussianHMM is None:
            raise ImportError(
                "hmmlearn is required for regime_method='hmm'. Install it first."
            )

        series = pd.Series(signal).astype(float)
        if input_kind == "daily_returns":
            mask = np.isfinite(series.values)
        else:
            mask = np.isfinite(series.values) & (series.values > 0)
        labels = pd.Series(index=series.index, dtype=object)

        n_components = int(self.config.get("hmm_n_components", 3))
        random_state = int(self.config.get("hmm_random_state", 42))

        if mask.sum() < max(30, 5 * n_components):
            return labels, None, {}

        if input_kind == "daily_returns":
            x = series.loc[mask].values.reshape(-1, 1)
        else:
            x = np.log(series.loc[mask]).values.reshape(-1, 1)

        transition_update_years = float(self.config.get("hmm_transition_update_years", 3.0))
        update_key: tuple[str, int] | None = None
        cached_transmat: np.ndarray | None = None
        if as_of_date is not None and transition_update_years > 0:
            half_year = 1 if as_of_date.month >= 7 else 0
            time_index = as_of_date.year + half_year * 0.5
            bucket_id = int(np.floor(time_index / transition_update_years))
            update_key = (input_kind, bucket_id)
            cached_transmat = self._hmm_transition_cache.get(update_key)

        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=int(self.config.get("hmm_n_iter", 1000)),
            random_state=random_state,
            init_params="mc",
            params="stmc" if cached_transmat is None else "smc",
        )
        model.startprob_ = np.array([0.33, 0.33, 0.34])
        model.transmat_ = (
            cached_transmat.copy()
            if cached_transmat is not None
            else np.array(
                [
                    [0.96, 0.03, 0.01],
                    [0.03, 0.94, 0.03],
                    [0.01, 0.03, 0.96],
                ]
            )
        )
        model.fit(x)

        if update_key is not None and cached_transmat is None:
            self._hmm_transition_cache[update_key] = np.array(model.transmat_, copy=True)

        states = model.predict(x)
        if input_kind == "daily_returns":
            covars = np.asarray(model.covars_).reshape(n_components, -1)[:, 0]
            order = np.argsort(np.sqrt(np.maximum(covars, 0.0)))
        else:
            state_means = model.means_.reshape(-1)
            order = np.argsort(state_means)
        state_to_label = {int(order[k]): REGIME_ORDER[k] for k in range(n_components)}
        labels.loc[mask] = pd.Series(states, index=series.loc[mask].index).map(state_to_label).values
        return labels, model, state_to_label

    def _metric_series(self, frame: pd.DataFrame, objective: str, ann_factor: float) -> pd.Series:
        if objective == "Sharpe":
            return _vectorized_sharpe(frame, ann_factor=ann_factor)
        if objective == "MaxDrawdown":
            return _vectorized_max_drawdown_loss(frame)
        if objective == "CVaR":
            return _vectorized_cvar_loss(frame)
        if objective == "Turnover":
            return frame.mean(axis=0, skipna=True).astype(float)
        raise ValueError(f"Unsupported objective: {objective}")

    def _selection_score(
        self,
        returns_frame: pd.DataFrame,
        weight_frame: pd.DataFrame,
        objective: str,
        ann_factor: float,
    ) -> pd.Series:
        avg_turnover = _vectorized_mean_turnover(weight_frame)
        if objective == "Sharpe":
            base = _vectorized_sharpe(returns_frame, ann_factor=ann_factor)
            turnover_cost = float(self.config.get("sharpe_turnover_cost", 0.5))
            return (base - turnover_cost * avg_turnover).astype(float)
        if objective == "MaxDrawdown":
            base = _vectorized_max_drawdown_loss(returns_frame)
            turnover_cost = float(self.config.get("maxdd_turnover_cost", 0.04))
            return (base + turnover_cost * avg_turnover).astype(float)
        return self._metric_series(returns_frame, objective=objective, ann_factor=ann_factor)

    def _pick_best(self, metric_values: pd.Series, objective: str) -> tuple[str | None, float]:
        clean = pd.Series(metric_values).replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return None, float("nan")

        if objective == "Sharpe":
            best_name = str(clean.idxmax())
        else:
            best_name = str(clean.idxmin())
        return best_name, float(clean.loc[best_name])

    def _pick_turnover_with_tiebreak(
        self,
        returns_frame: pd.DataFrame,
        turnover_frame: pd.DataFrame,
        ann_factor: float,
    ) -> tuple[str | None, float]:
        # This could be the best strategy we found:
        # minimize turnover first, then prefer lower max drawdown,
        # and finally prefer higher Sharpe within small tie ranges.
        turnover_values = self._metric_series(
            turnover_frame, objective="Turnover", ann_factor=ann_factor
        )
        clean_turnover = pd.Series(turnover_values).replace([np.inf, -np.inf], np.nan).dropna()
        if clean_turnover.empty:
            return None, float("nan")

        turnover_tolerance = float(self.config.get("turnover_reasonable_range", 0.0025))
        maxdd_tolerance = float(self.config.get("turnover_maxdd_small_range", 0.01))

        min_turnover = float(clean_turnover.min())
        turnover_candidates = clean_turnover[
            clean_turnover <= (min_turnover + turnover_tolerance)
        ]
        if len(turnover_candidates) == 1:
            best_name = str(turnover_candidates.idxmin())
            return best_name, float(turnover_candidates.loc[best_name])

        candidate_cols = list(turnover_candidates.index)
        candidate_returns = returns_frame.loc[:, candidate_cols]
        candidate_maxdd = _vectorized_max_drawdown_loss(candidate_returns).dropna()
        if candidate_maxdd.empty:
            best_name = str(turnover_candidates.idxmin())
            return best_name, float(turnover_candidates.loc[best_name])

        min_maxdd = float(candidate_maxdd.min())
        maxdd_candidates = candidate_maxdd[candidate_maxdd <= (min_maxdd + maxdd_tolerance)]
        if len(maxdd_candidates) == 1:
            best_name = str(maxdd_candidates.idxmin())
            return best_name, float(clean_turnover.loc[best_name])

        finalist_cols = list(maxdd_candidates.index)
        finalist_returns = returns_frame.loc[:, finalist_cols]
        finalist_sharpe = _vectorized_sharpe(finalist_returns, ann_factor=ann_factor).dropna()
        if finalist_sharpe.empty:
            best_name = str(maxdd_candidates.idxmin())
            return best_name, float(clean_turnover.loc[best_name])

        best_name = str(finalist_sharpe.idxmax())
        return best_name, float(clean_turnover.loc[best_name])

    def _build_winner_table(
        self,
        train_returns: pd.DataFrame,
        train_regime_signal: pd.Series,
        ann_factor: float,
        as_of_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, float, float, str | None]:
        signal_clean = train_regime_signal.dropna()
        if len(signal_clean) < int(self.config.get("min_train_proxy_obs", 252)):
            return pd.DataFrame(), float("nan"), float("nan"), None

        regime_method = str(self.config.get("regime_method", "proxy_quantile")).lower()
        q_low = float("nan")
        q_high = float("nan")
        current_regime = None

        if regime_method == "hmm":
            hmm_input = str(self.config.get("hmm_input", "intraday_vol")).lower()
            train_regime, _, _ = self._fit_hmm_regimes(
                train_regime_signal,
                input_kind=hmm_input,
                as_of_date=as_of_date,
            )
            current_regime = train_regime.loc[as_of_date] if as_of_date in train_regime.index else None
        else:
            q_low = float(signal_clean.quantile(1.0 / 3.0))
            q_high = float(signal_clean.quantile(2.0 / 3.0))
            train_regime = train_regime_signal.apply(
                lambda x: self._classify_regime(float(x), q_low, q_high) if np.isfinite(x) else None
            )
            current_regime = (
                self._classify_regime(float(train_regime_signal.loc[as_of_date]), q_low, q_high)
                if as_of_date in train_regime_signal.index
                and np.isfinite(train_regime_signal.loc[as_of_date])
                else None
            )

        rows: list[dict[str, object]] = []
        for regime in REGIME_ORDER:
            regime_mask = train_regime.eq(regime)
            frame = train_returns.loc[regime_mask].copy()
            frame = frame.dropna(axis=1, how="all")
            if frame.empty:
                continue

            weight_frame = self.weight_panel.loc[frame.index, frame.columns].astype(float)

            sharpe_values = self._selection_score(
                returns_frame=frame,
                weight_frame=weight_frame,
                objective="Sharpe",
                ann_factor=ann_factor,
            )
            sharpe_best_strategy, sharpe_best_value = self._pick_best(
                sharpe_values, objective="Sharpe"
            )
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "regime": regime,
                    "objective": "Sharpe",
                    "best_strategy": sharpe_best_strategy,
                    "best_metric_value": sharpe_best_value,
                    "n_obs": int(regime_mask.sum()),
                    "q_low": q_low,
                    "q_high": q_high,
                }
            )

            for objective in ["MaxDrawdown", "CVaR"]:
                metric_values = self._selection_score(
                    returns_frame=frame,
                    weight_frame=weight_frame,
                    objective=objective,
                    ann_factor=ann_factor,
                )
                best_strategy, best_value = self._pick_best(
                    metric_values, objective=objective
                )
                rows.append(
                    {
                        "as_of_date": as_of_date,
                        "regime": regime,
                        "objective": objective,
                        "best_strategy": best_strategy,
                        "best_metric_value": best_value,
                        "n_obs": int(regime_mask.sum()),
                        "q_low": q_low,
                        "q_high": q_high,
                    }
                )

            turnover_frame = weight_frame.diff().abs().fillna(0.0)
            turnover_best_strategy, turnover_best_value = self._pick_turnover_with_tiebreak(
                returns_frame=frame,
                turnover_frame=turnover_frame,
                ann_factor=ann_factor,
            )
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "regime": regime,
                    "objective": "Ultimate",
                    "best_strategy": turnover_best_strategy,
                    "best_metric_value": turnover_best_value,
                    "n_obs": int(regime_mask.sum()),
                    "q_low": q_low,
                    "q_high": q_high,
                }
            )

        return pd.DataFrame(rows), q_low, q_high, current_regime

    def run(self, mode: str = "all"):
        if mode != "all":
            raise ValueError("RollingRegimeMixEngine currently supports mode='all' only.")

        ann_factor = float(self.config.get("ann_factor", 252.0))
        train_years = int(self.config.get("train_years", 6))
        panel, weight_panel = self._load_strategy_data()
        regime_method = str(self.config.get("regime_method", "proxy_quantile")).lower()
        hmm_input = str(self.config.get("hmm_input", "intraday_vol")).lower()
        if regime_method == "hmm" and hmm_input == "daily_returns":
            regime_signal = self._load_daily_returns_signal().reindex(panel.index)
        else:
            regime_signal = self._load_intraday_proxy().reindex(panel.index)

        common_index = panel.index.intersection(regime_signal.dropna().index)
        panel = panel.reindex(common_index).sort_index()
        weight_panel = weight_panel.reindex(common_index).sort_index()
        regime_signal = regime_signal.reindex(common_index).sort_index()
        self.weight_panel = weight_panel

        if len(common_index) < 252 * (train_years + 1):
            raise ValueError(
                f"Not enough history for a {train_years}-year training window. "
                f"Only {len(common_index)} aligned observations were found."
            )

        first_date = common_index.min()
        oos_start = first_date + pd.DateOffset(years=train_years)
        decision_dates = common_index[(common_index >= oos_start) & (common_index < common_index.max())]

        if decision_dates.empty:
            raise ValueError("No out-of-sample dates remain after the initial training window.")

        result_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []
        initial_winners: pd.DataFrame | None = None

        for decision_date in decision_dates:
            train_start = decision_date - pd.DateOffset(years=train_years)
            train_mask = (panel.index >= train_start) & (panel.index <= decision_date)

            winner_table, q_low, q_high, current_regime = self._build_winner_table(
                train_returns=panel.loc[train_mask],
                train_regime_signal=regime_signal.loc[train_mask],
                ann_factor=ann_factor,
                as_of_date=decision_date,
            )
            if winner_table.empty:
                continue

            if initial_winners is None:
                initial_winners = winner_table.copy()

            if current_regime is None:
                continue

            loc = common_index.get_loc(decision_date)
            if loc >= len(common_index) - 1:
                continue

            realized_date = common_index[loc + 1]
            row: dict[str, object] = {
                "decision_date": decision_date,
                "regime_signal_value": float(regime_signal.loc[decision_date]),
                "regime_method": regime_method,
                "hmm_input": hmm_input if regime_method == "hmm" else "",
                "q_low": q_low,
                "q_high": q_high,
                "predicted_regime": current_regime,
            }

            for objective in OBJECTIVE_ORDER:
                selected_row = winner_table[
                    (winner_table["objective"] == objective)
                    & (winner_table["regime"] == current_regime)
                ]
                if selected_row.empty:
                    row[f"{objective}_selected_strategy"] = None
                    row[f"{objective}_return"] = np.nan
                    row[f"{objective}_log_return"] = np.nan
                    continue

                selected_strategy = selected_row["best_strategy"].iloc[0]
                realized_return = panel.at[realized_date, selected_strategy]
                realized_weight = weight_panel.at[realized_date, selected_strategy]

                row[f"{objective}_selected_strategy"] = selected_strategy
                row[f"{objective}_return"] = float(realized_return)
                row[f"{objective}_log_return"] = float(np.log1p(realized_return))
                row[f"{objective}_weight"] = (
                    float(realized_weight) if np.isfinite(realized_weight) else np.nan
                )

                selection_rows.append(
                    {
                        "decision_date": decision_date,
                        "realized_date": realized_date,
                        "objective": objective,
                        "predicted_regime": current_regime,
                        "selected_strategy": selected_strategy,
                        "selected_metric_value": float(selected_row["best_metric_value"].iloc[0]),
                        "q_low": q_low,
                        "q_high": q_high,
                    }
                )

            row["realized_date"] = realized_date
            result_rows.append(row)

        if not result_rows:
            raise ValueError("The rolling regime engine did not produce any out-of-sample rows.")

        result = pd.DataFrame(result_rows).set_index("realized_date").sort_index()
        result.index.name = "date"

        for objective in OBJECTIVE_ORDER:
            simple_col = f"{objective}_return"
            log_col = f"{objective}_log_return"
            eq_col = f"{objective}_equity_curve"
            weight_col = f"{objective}_weight"
            turnover_col = f"{objective}_turnover"
            result[log_col] = _simple_to_log(result[simple_col].fillna(0.0))
            result[eq_col] = float(self.config.get("initial_capital", 1000.0)) * np.exp(
                result[log_col].fillna(0.0).cumsum()
            )
            result[turnover_col] = result[weight_col].astype(float).diff().abs().fillna(0.0)

        summary_rows: list[dict[str, object]] = []
        for objective in OBJECTIVE_ORDER:
            returns = result[f"{objective}_return"].dropna()
            turnover = result[f"{objective}_turnover"].dropna()
            summary_rows.append(
                {
                    "CombinedStrategy": objective,
                    "Sharpe": _sharpe_ratio(returns, ann_factor=ann_factor),
                    "MaxDrawdown": _max_drawdown_loss(returns),
                    "CVaR": _cvar_loss(returns),
                    "AnnualizedVol": _annualized_vol(returns, ann_factor=ann_factor),
                    "AnnualizedReturn": _annualized_return(returns, ann_factor=ann_factor),
                    "AverageTurnover": float(turnover.mean()) if not turnover.empty else float("nan"),
                    "AnnualizedTurnover": float(turnover.mean() * ann_factor)
                    if not turnover.empty
                    else float("nan"),
                    "Days": int(len(returns)),
                }
            )

        self.result = result
        self.summary_table = pd.DataFrame(summary_rows)
        self.initial_winners = initial_winners
        self.selection_history = pd.DataFrame(selection_rows)
        return self.result

    def save(self):
        if self.result is None or self.summary_table is None:
            raise ValueError("No results to save. Run the engine first.")

        results_dir = Env.path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        result_path = results_dir / f"{self.name}.parquet"
        summary_path = results_dir / f"{self.name}_summary.csv"
        initial_winners_path = results_dir / f"{self.name}_initial_winners.csv"
        selection_history_path = results_dir / f"{self.name}_selection_history.csv"

        self.result.to_parquet(result_path)
        self.summary_table.to_csv(summary_path, index=False)

        if self.initial_winners is not None:
            self.initial_winners.to_csv(initial_winners_path, index=False)
        if self.selection_history is not None:
            self.selection_history.to_csv(selection_history_path, index=False)

        return {
            "result_path": result_path,
            "summary_path": summary_path,
            "initial_winners_path": initial_winners_path,
            "selection_history_path": selection_history_path,
        }

    def summary(self):
        if self.summary_table is None:
            print("No results. Run the engine first.")
            return

        print("\nRolling Regime Mix Summary")
        print(self.summary_table.to_string(index=False))
