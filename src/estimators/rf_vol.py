import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from src.estimators.base import Estimator
from src.env import Env

def _load_vix() -> pd.Series | None:
    """Return a DatetimeIndex-aligned VIX close series, or None."""
    try:
        path = Env.path("processed") / "VIX_Daily_Processed.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        for col in ("close", "Close", "vix", "VIX", "vix_close"):
            if col in df.columns:
                return df[col].astype(float).sort_index()
        num = df.select_dtypes(include="number")
        return num.iloc[:, 0].sort_index() if len(num.columns) else None
    except Exception:
        return None


def _load_raw_csv() -> pd.DataFrame | None:
    """Return the raw prices CSV as a DataFrame, or None."""
    try:
        path = Env.path("raw") / "prices.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return None

class RandomForestVolEstimator(Estimator):

    def __init__(self, params=None):
        super().__init__(params)
        self.n_estimators  = int(self.params.get("n_estimators", 100))
        self.max_depth     = self.params.get("max_depth", 10)
        if self.max_depth is not None:
            self.max_depth = int(self.max_depth)
        self.lags          = self.params.get("lags", [1, 5, 22])
        self.train_window  = int(self.params.get("train_window", 252))
        self.rv_window     = int(self.params.get("rv_window", 5))
        self.use_hmm       = bool(self.params.get("use_hmm", True))
        self.n_iter        = int(self.params.get("n_iter", 3))
        self.tune_period   = int(self.params.get("tune_period", 21))
        self.refit_every   = max(1, int(self.params.get("refit_every", 21)))
        self.n_jobs        = int(self.params.get("n_jobs", 1))
        self.random_state  = int(self.params.get("random_state", 42))
        self.hmm_model     = None
        self.hmm_update_days = 22
        self.auto_tune     = bool(self.params.get("auto_tune", True))
        self.cv_splits     = int(self.params.get("cv_splits", 3))
        self._vix: pd.Series | None       = None
        self._raw_df: pd.DataFrame | None = None
        self._ext_loaded: bool            = False
        self._best_model  = None
        self._model_fit_count = None
        self._count       = 0
        self._cache_end = None
        self._cache_len = None
        self._cache_vol = None
        self._oos_log: list[dict] = []

        # ── Regime Sharpe tracking ────────────────────────────────────
        # 每次 estimate_window 呼叫後，把「最後一天」的資料存進來
        # 格式: list of {"date": Timestamp, "regime": int, "log_return": float}
        self._regime_log: list[dict] = []

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _load_ext(self):
        if self._ext_loaded:
            return
        self._vix    = _load_vix()
        self._raw_df = _load_raw_csv()
        self._ext_loaded = True

    def _get_hmm_regimes(self, log_vol: pd.Series) -> pd.Series:
        """
        HMM regime detection.
        0 = 低波動 (Low)  |  1 = 中波動 (Mid)  |  2 = 高波動 (High)
        """
        from hmmlearn.hmm import GaussianHMM
        vals = log_vol.values.reshape(-1, 1)
        mask = np.isfinite(vals).flatten()

        if self.hmm_model is None or self._count % self.hmm_update_days == 0:
            X_train = vals[mask]
            if len(X_train) < 50:
                return pd.Series(np.nan, index=log_vol.index)

            model = GaussianHMM(n_components=3, covariance_type="diag",
                                n_iter=100, random_state=42)
            model.fit(X_train)

            order = np.argsort(model.means_.flatten())
            self.hmm_remap = {int(order[k]): k for k in range(3)}
            self.hmm_model = model

        raw_states     = self.hmm_model.predict(vals[mask])
        ordered_states = np.vectorize(self.hmm_remap.get)(raw_states)

        out = pd.Series(np.nan, index=log_vol.index)
        out.iloc[np.where(mask)[0]] = ordered_states.astype(float)
        return out

    def _build_feature_df(self, window_returns: pd.Series) -> pd.DataFrame:
        self._load_ext()
        r   = window_returns.dropna().astype(float)
        idx = r.index

        rv = r.rolling(window=self.rv_window).std() * np.sqrt(252)
        df = pd.DataFrame({"target": rv}, index=idx)

        for lag in self.lags:
            df[f"lag_{lag}"] = df["target"].shift(lag)

        df["log_return"] = r

        log_price = r.cumsum()
        roll_max  = log_price.rolling(window=5).max()
        roll_min  = log_price.rolling(window=5).min()
        df["hl_range"] = (roll_max - roll_min).clip(lower=0)

        if self._vix is not None:
            vix_aligned  = self._vix.reindex(idx, method="ffill")
            df["vix"]    = vix_aligned.values
            df["vix_sq"] = vix_aligned.values ** 2

        if self._raw_df is not None and "rv20" in self._raw_df.columns:
            rv20 = (
                pd.to_numeric(self._raw_df["rv20"], errors="coerce")
                .reindex(idx, method="ffill")
            )
            df["rv20_lag1"] = rv20.shift(1).values
        else:
            df["rv20_lag1"] = (
                r.rolling(window=20).std() * np.sqrt(252)
            ).shift(1).values

        if self._raw_df is not None and "volume" in self._raw_df.columns:
            vol_series = (
                pd.to_numeric(self._raw_df["volume"], errors="coerce")
                .reindex(idx, method="ffill")
            )
            vol_ma20        = vol_series.rolling(window=20).mean()
            df["vol_spike"] = (vol_series > 2 * vol_ma20).astype(float).values

        if self.use_hmm:
            log_vol          = np.log(rv.clip(lower=1e-8))
            df["hmm_regime"] = self._get_hmm_regimes(log_vol).values

        return df

    def _tune_hyperparameters(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        param_distributions = {
            "n_estimators":    [50, 100, 200],
            "max_depth":       [5, 10, 15],
            "min_samples_leaf":[1, 2, 4],
            "max_features":    ["sqrt", None],
        }
        rf     = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        search.fit(X, y)
        return search.best_estimator_
    
    REGIME_NAMES = {0: "Low Vol", 1: "Mid Vol", 2: "High Vol"}

    def compute_regime_sharpe(
        self,
        annualize: bool = True,
        risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        計算每個 HMM regime 的 Sharpe Ratio。

        Parameters
        ----------
        annualize : bool
            True  → Sharpe = (mean - rf) / std * sqrt(252)
            False → 不年化，直接用日頻均值與標準差
        risk_free_rate : float
            年化無風險利率（預設 0）。日頻版會自動除以 252。

        Returns
        -------
        pd.DataFrame
            columns: regime | regime_name | n_days |
                     mean_daily_return | std_daily_return | sharpe_ratio
        """
        if not self._regime_log:
            raise ValueError("No regime data recorded yet. Run estimate_window() first.")

        log_df = pd.DataFrame(self._regime_log)

        rf_daily = risk_free_rate / 252  # 轉換成日頻無風險報酬

        records = []
        for regime_id in sorted(log_df["regime"].unique()):
            subset = log_df.loc[log_df["regime"] == regime_id, "log_return"]
            n      = len(subset)
            mu     = subset.mean()        # 日均 log return
            sigma  = subset.std(ddof=1)   # 日標準差

            if sigma == 0 or not np.isfinite(sigma):
                sharpe = np.nan
            elif annualize:
                # 年化：(日均超額報酬) / (日標準差) * sqrt(252)
                sharpe = (mu - rf_daily) / sigma * np.sqrt(252)
            else:
                sharpe = (mu - rf_daily) / sigma

            records.append({
                "regime":            regime_id,
                "regime_name":       self.REGIME_NAMES.get(regime_id, f"Regime {regime_id}"),
                "n_days":            n,
                "mean_daily_return": round(mu, 6),
                "std_daily_return":  round(sigma, 6),
                "sharpe_ratio":      round(sharpe, 4) if np.isfinite(sharpe) else np.nan,
            })

        return pd.DataFrame(records).set_index("regime")

    def regime_sharpe_summary(self, **kwargs) -> None:
        """Pretty-print the regime Sharpe table."""
        df = self.compute_regime_sharpe(**kwargs)
        print("\n=== Regime Sharpe Ratio Summary ===")
        print(df.to_string())
        print()

    def compute_oos_r2(self) -> pd.DataFrame:
        if not self._oos_log:
            raise ValueError("No OOS data recorded yet.")
        if not self._regime_log:
            raise ValueError("No regime data recorded yet.")

        oos_df    = pd.DataFrame(self._oos_log).set_index("date")
        regime_df = pd.DataFrame(self._regime_log).set_index("date")

        # 合併 regime 標籤
        merged = oos_df.join(regime_df[["regime"]], how="inner")

        # Overall OOS R²
        merged["benchmark"] = merged["actual"].expanding().mean().shift(1)
        merged = merged.dropna()

        def _r2(df):
            ss_res   = ((df["actual"] - df["predicted"]) ** 2).sum()
            ss_bench = ((df["actual"] - df["benchmark"]) ** 2).sum()
            return 1 - ss_res / ss_bench if ss_bench != 0 else np.nan

        print("\n=== OOS R² by Regime ===")
        records = []

        # Overall
        r2_all = _r2(merged)
        print(f"{'Overall':10s} | n={len(merged):4d} | OOS R² = {r2_all:.4f}")
        records.append({"regime": "Overall", "n_days": len(merged), "oos_r2": r2_all})

        # Per regime
        for regime_id, name in self.REGIME_NAMES.items():
            subset = merged[merged["regime"] == regime_id]
            if len(subset) < 5:
                continue
            r2 = _r2(subset)
            print(f"{name:10s} | n={len(subset):4d} | OOS R² = {r2:.4f}")
            records.append({"regime": name, "n_days": len(subset), "oos_r2": r2})

        return pd.DataFrame(records).set_index("regime")

    # ─────────────────────────────────────────────────────────────────
    # Main estimation
    # ─────────────────────────────────────────────────────────────────

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = pd.Series(window_returns).dropna().astype(float)
        end_ts = pd.Timestamp(r.index[-1]) if len(r) else None
        if (
            end_ts is not None
            and self._cache_end == end_ts
            and self._cache_len == len(r)
            and self._cache_vol is not None
        ):
            return float(self._cache_vol)

        min_required = max(self.lags) + self.rv_window + 20

        if len(r) < min_required:
            return float(r.std() * np.sqrt(252)) if len(r) > 1 else 0.20

        try:
            df = self._build_feature_df(r)
            df = df.dropna()

            if len(df) < 20:
                return float(r.std() * np.sqrt(252))

            train        = df.iloc[-self.train_window:]
            feature_cols = [c for c in train.columns if c != "target"]
            X_train      = train[feature_cols].values
            y_train      = train["target"].values

            should_refit = (
                self._best_model is None
                or self._model_fit_count is None
                or self._count - self._model_fit_count >= self.refit_every
            )

            if self.auto_tune:
                if should_refit and (self._best_model is None or self._count % self.tune_period == 0):
                    self._best_model = self._tune_hyperparameters(X_train, y_train)
                    self._model_fit_count = self._count
                elif should_refit:
                    self._best_model.fit(X_train, y_train)
                    self._model_fit_count = self._count
            else:
                if self._best_model is None:
                    self._best_model = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )
                if should_refit:
                    self._best_model.fit(X_train, y_train)
                    self._model_fit_count = self._count

            self._count += 1

            X_next     = df[feature_cols].iloc[[-1]].values
            prediction = float(self._best_model.predict(X_next)[0])

            # ── 記錄最後一天的 regime & log_return ──────────────────
            if self.use_hmm and "hmm_regime" in df.columns:
                last_row = df.iloc[-1]
                regime   = last_row["hmm_regime"]
                if np.isfinite(regime):
                    self._regime_log.append({
                        "date":       df.index[-1],
                        "regime":     int(regime),
                        "log_return": float(last_row["log_return"]),
                    })

            if not np.isfinite(prediction) or prediction <= 0:
                return float(r.std() * np.sqrt(252))
            
            actual_rv = float(df["target"].iloc[-1])  # 當天實際 RV
            self._oos_log.append({
                "date":       df.index[-1],
                "predicted":  prediction,
                "actual":     actual_rv,
            })

            self._cache_end = end_ts
            self._cache_len = len(r)
            self._cache_vol = prediction
            return prediction

        except Exception as e:
            print(f"Estimation error: {e}")
            return float(r.std() * np.sqrt(252))
