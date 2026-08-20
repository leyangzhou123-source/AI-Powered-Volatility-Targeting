"""
XGBoost Volatility Estimator — Master Dataset Edition
======================================================
Improvements over the previous version:
  1.  Single data source: loads everything from Master_Dataset.parquet
      (vix_close, t10y2y, t10y3m, rv20, returns_clean) — no separate VIX parquet.
  2.  Yield-curve slope features (t10y2y, t10y3m) and their momentum / inversion
      signals — empirically, curve slope leads equity vol with a lag.
  3.  Pre-computed rv20 column used directly as a higher-quality realized-variance
      feature (rolling 20-day, databento-sourced) instead of r^2 rolling mean.
  4.  returns_clean is used for the in-window RV construction so that futures
      roll days don't inflate daily squared returns.
  5.  Jump indicator flags days where |r| > 3σ (rolling), helping XGBoost learn
      that jumps and regime changes are transient.
  6.  Time-decay sample weights: recent observations get more weight in training,
      reducing stale-regime influence.
  7.  Feature names remain consistent (no dtype mismatch issues after model.fit).

Engine interface
----------------
  estimate_window(window_returns: pd.Series) -> float   ← engine calls this
  estimate(t, returns)                                  ← fallback
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from src.estimators.base import Estimator
from src.env import Env


class XGB_VIX(Estimator):
    """
    XGBoost volatility estimator with enriched HAR + VIX + yield-curve features.
    Trains from scratch on every rolling window (expanding-window via roll_window=1000
    in the YAML).  Target is log-variance; output is annualised volatility.
    """

    # ------------------------------------------------------------------ #
    # init                                                                #
    # ------------------------------------------------------------------ #
    def __init__(self, params=None):
        super().__init__(params)
        self.w_lookback      = int(self.params.get("w_lookback", 5))
        self.m_lookback      = int(self.params.get("m_lookback", 22))
        self.min_obs         = int(self.params.get("min_obs", 120))
        self.ann             = float(self.params.get("vol_ann", 252.0))
        self.rv20_is_annualized_vol = self.params.get("rv20_is_annualized_vol")

        # XGBoost hyper-parameters (Optuna-tuned defaults kept from prior version)
        self.n_estimators    = int(self.params.get("n_estimators", 1000))
        self.max_depth       = int(self.params.get("max_depth", 2))
        self.learning_rate   = float(self.params.get("learning_rate", 0.009586))
        self.subsample       = float(self.params.get("subsample", 0.7259))
        self.colsample_bytree= float(self.params.get("colsample_bytree", 0.7117))
        self.reg_lambda      = float(self.params.get("reg_lambda", 0.000897))
        self.reg_alpha       = float(self.params.get("reg_alpha", 0.0000037))
        self.min_child_weight= int(self.params.get("min_child_weight", 4))
        # Time-decay weight factor (per step). 0.999 ≈ 2-year effective half-life
        self.decay_factor    = float(self.params.get("decay_factor", 0.999))
        self.random_state    = int(self.params.get("random_state", 42))
        self.refit_every     = max(1, int(self.params.get("refit_every", 21)))
        self._model          = None
        self._model_fit_count = None
        self._cache_end      = None
        self._cache_len      = None
        self._cache_vol      = None

        # Engine diagnostics
        self.diag = {
            "fallback_ewma":          0,
            "fallback_ewma_ts":       [],
            "fallback_rv":            0,
            "fallback_rv_ts":         [],
            "non_converge":           0,
            "non_converge_ts":        [],
            "warmup_insufficient":    0,
            "warmup_insufficient_ts": [],
        }

        # ------------------------------------------------------------------
        # Load auxiliary data from Master_Dataset
        # ------------------------------------------------------------------
        self._master_path = Env.path("processed") / "Master_Dataset.parquet"
        self._daily_path = Env.path("processed") / "ES_Daily_Processed.parquet"
        self._master: pd.DataFrame | None = None
        self._daily: pd.DataFrame | None = None
        self._load_master()
        self._load_daily()

    # ------------------------------------------------------------------ #
    # Data loading helpers                                                #
    # ------------------------------------------------------------------ #
    def _load_master(self) -> None:
        """Load Master_Dataset once at init; extract and normalise all auxiliary series."""
        if not self._master_path.exists():
            print(f"⚠️  XGB_VIX: Master_Dataset not found at {self._master_path}. "
                  "Auxiliary features (VIX, rates) will be zeroed out.")
            self._master = None
            return

        df = pd.read_parquet(self._master_path)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df.sort_index()
        self._master = df

    def _load_daily(self) -> None:
        """Load the strategy daily dataset for returns_clean fallback."""
        if not self._daily_path.exists():
            self._daily = None
            return

        df = pd.read_parquet(self._daily_path)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df.sort_index()
        self._daily = df

    @staticmethod
    def _align_series(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Align an auxiliary series to `dates` without look-ahead bias.
        Dates are normalised; forward-fill then backward-fill for gaps.
        """
        dates_norm = pd.to_datetime(dates).tz_localize(None).normalize()
        s = series.copy()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        aligned = s.reindex(dates_norm).ffill().bfill()
        aligned.index = dates
        return aligned

    def _get_aux(self, col: str, dates: pd.DatetimeIndex,
                 fill_value: float = 0.0) -> pd.Series:
        """
        Safe accessor: returns aligned series for `col` from master dataset.
        Returns fill_value if master not loaded or column absent.
        """
        if self._master is None or col not in self._master.columns:
            return pd.Series(fill_value, index=dates)
        return self._align_series(self._master[col].astype(float), dates)

    def _get_returns_clean(self, dates: pd.DatetimeIndex, fallback: pd.Series) -> pd.Series:
        """Prefer Master_Dataset returns_clean, then ES_Daily_Processed, then window returns."""
        r_clean = self._get_aux("returns_clean", dates, fill_value=np.nan)
        if r_clean.notna().any():
            return r_clean.where(r_clean.notna(), fallback)

        if self._daily is not None and "returns_clean" in self._daily.columns:
            daily_clean = self._align_series(self._daily["returns_clean"].astype(float), dates)
            return daily_clean.where(daily_clean.notna(), fallback)

        return fallback

    def _rv20_in_daily_variance(self, rv20_raw: pd.Series) -> pd.Series:
        """
        Normalize rv20 into daily variance units.
        If rv20 is already a small variance-like series, leave it unchanged.
        If it looks like annualized volatility in decimal terms, square and de-annualize.
        """
        rv20 = rv20_raw.astype(float).copy()
        valid = rv20.dropna()
        if valid.empty:
            return rv20

        if self.rv20_is_annualized_vol is None:
            median_level = float(valid.median())
            looks_annualized_vol = median_level > 0.01
        else:
            looks_annualized_vol = bool(self.rv20_is_annualized_vol)

        if looks_annualized_vol:
            return rv20.pow(2) / self.ann
        return rv20

    # ------------------------------------------------------------------ #
    # Feature engineering                                                 #
    # ------------------------------------------------------------------ #
    def _build_features(self, r: pd.Series, shift_for_training: bool = True) -> pd.DataFrame:
        """
        Build the full feature matrix from a window of returns.
        All features are shifted by 1 day so there is zero look-ahead bias.

        Returns a DataFrame of shape (n_rows, n_features).
        The last row represents "features known as of yesterday → forecast today".
        """
        dates = r.index

        # ── 1. Returns-based RV features from tradable strategy returns ──
        rv_d  = r.pow(2)
        rv_w  = rv_d.rolling(self.w_lookback).mean()
        rv_m  = rv_d.rolling(self.m_lookback).mean()
        rv_q  = rv_d.rolling(63).mean()                  # quarterly horizon

        # ── 2. Pre-computed RV20 from data pipeline (higher quality) ──
        rv20_raw  = self._get_aux("rv20", dates, fill_value=np.nan)
        rv20_raw = self._rv20_in_daily_variance(rv20_raw)
        rv20_feat = rv20_raw.fillna(rv_m)

        # ── 3. Structural / behavioural signals ──
        ret_sign    = (r < 0).astype(int)                # leverage-effect indicator
        vol_of_vol  = rv_d.rolling(self.m_lookback).std()# regime instability proxy
        roll_sigma  = r.rolling(self.m_lookback).std()
        jump_flag   = (r.abs() > 3.0 * roll_sigma).astype(int)  # jump indicator

        # ── 4. VIX features ──
        vix_raw      = self._get_aux("vix_close", dates, fill_value=np.nan)
        # Convert VIX (annualised vol %) to daily variance units
        vix_var_d    = (vix_raw / 100.0) ** 2 / self.ann
        vix_diff     = vix_var_d.diff()                  # VIX momentum
        vrp          = vix_var_d / (rv_w + 1e-8)         # variance risk premium proxy

        # ── 5. Yield-curve slope features ──
        # t10y2y = 10yr minus 2yr (positive = normal, negative = inverted)
        # t10y3m = 10yr minus 3m (tighter link to Fed policy expectations)
        slope_2y10y  = self._get_aux("t10y2y", dates, fill_value=np.nan)
        slope_3m10y  = self._get_aux("t10y3m", dates, fill_value=np.nan)

        d_slope_2y10y  = slope_2y10y.diff()              # slope momentum
        d_slope_3m10y  = slope_3m10y.diff()
        accel_slope    = d_slope_2y10y.diff()             # slope acceleration (2nd diff)

        # Inversion dummy: 1 when 2-10yr curve is inverted
        inversion_2y10y = (slope_2y10y < 0).astype(float)
        inversion_3m10y = (slope_3m10y < 0).astype(float)

        # Interaction: VRP amplified by curve inversion (stress signal)
        vrp_x_inv     = vrp * inversion_2y10y
        # Interaction: slope level × vol-of-vol (macro uncertainty)
        slope_x_vov   = slope_2y10y * vol_of_vol

        # ── 6. Assemble features ──
        features = pd.concat([
            rv_d.rename("rv_d"),
            rv_w.rename("rv_w"),
            rv_m.rename("rv_m"),
            rv_q.rename("rv_q"),
            rv20_feat.rename("rv20"),
            ret_sign.rename("ret_sign"),
            vol_of_vol.rename("vol_of_vol"),
            jump_flag.rename("jump_flag"),
            vix_var_d.rename("vix"),
            vix_diff.rename("vix_diff"),
            vrp.rename("vrp"),
            slope_2y10y.rename("slope_2y10y"),
            slope_3m10y.rename("slope_3m10y"),
            d_slope_2y10y.rename("d_slope_2y10y"),
            d_slope_3m10y.rename("d_slope_3m10y"),
            accel_slope.rename("accel_slope"),
            inversion_2y10y.rename("inv_2y10y"),
            inversion_3m10y.rename("inv_3m10y"),
            vrp_x_inv.rename("vrp_x_inv"),
            slope_x_vov.rename("slope_x_vov"),
        ], axis=1)

        if shift_for_training:
            features = features.shift(1)  # train y_t on information known at t-1

        return features

    # ------------------------------------------------------------------ #
    # Core estimator                                                      #
    # ------------------------------------------------------------------ #
    def estimate_window(self, window_returns: pd.Series) -> float:
        """
        Called by the engine on every rolling step.
        window_returns: pd.Series with DatetimeIndex, length == roll_window.
        Returns: annualised vol forecast (float).
        """
        r      = pd.Series(window_returns).dropna().astype(float)
        end_ts = pd.Timestamp(r.index[-1]) if len(r) else pd.NaT
        if (
            pd.notna(end_ts)
            and self._cache_end == end_ts
            and self._cache_len == len(r)
            and self._cache_vol is not None
        ):
            return float(self._cache_vol)

        # Warm-up guard: need 63 days for the quarterly RV feature
        min_needed = max(63, self.m_lookback + 2)
        if len(r) < min_needed:
            self.diag["warmup_insufficient"] += 1
            if pd.notna(end_ts):
                self.diag["warmup_insufficient_ts"].append(end_ts)
            return float("nan")

        # ── Build feature matrix and realized-volatility target ──
        X_all = self._build_features(r, shift_for_training=True)
        X_pred_all = self._build_features(r, shift_for_training=False)
        y_all = (
            r.rolling(self.m_lookback).std(ddof=1) * np.sqrt(self.ann)
        ).rename("y")

        m = pd.concat([X_all, y_all], axis=1).dropna()

        # Minimum obs guard after NaN-drop (warm-up for rolling features)
        if len(m) < self.min_obs:
            self.diag["fallback_rv"] += 1
            if pd.notna(end_ts):
                self.diag["fallback_rv_ts"].append(end_ts)
            var_hat = float(r.pow(2).tail(self.m_lookback).mean())
            return float(np.sqrt(max(var_hat, 1e-18)) * np.sqrt(self.ann))

        feat_cols = [c for c in m.columns if c != "y"]
        X_train   = m[feat_cols].values
        y_train   = m["y"].values

        # Forecast next-day variance using the latest observable feature state.
        X_pred_df = X_pred_all[feat_cols].iloc[[-1]].dropna()
        if X_pred_df.empty:
            self.diag["fallback_rv"] += 1
            if pd.notna(end_ts):
                self.diag["fallback_rv_ts"].append(end_ts)
            var_hat = float(r.pow(2).tail(self.m_lookback).mean())
            return float(np.sqrt(max(var_hat, 1e-18)) * np.sqrt(self.ann))
        X_pred = X_pred_df.values

        # ── Time-decay sample weights ──
        n_train  = len(X_train)
        weights  = np.power(self.decay_factor, np.arange(n_train)[::-1])

        try:
            should_refit = (
                self._model is None
                or self._model_fit_count is None
                or len(r) - self._model_fit_count >= self.refit_every
            )
            if should_refit:
                self._model = xgb.XGBRegressor(
                    n_estimators     = self.n_estimators,
                    max_depth        = self.max_depth,
                    learning_rate    = self.learning_rate,
                    subsample        = self.subsample,
                    colsample_bytree = self.colsample_bytree,
                    reg_lambda       = self.reg_lambda,
                    reg_alpha        = self.reg_alpha,
                    min_child_weight = self.min_child_weight,
                    objective        = "reg:squarederror",
                    n_jobs           = 1,   # prevent thread thrashing inside rolling loop
                    verbosity        = 0,
                    random_state     = self.random_state,
                )
                self._model.fit(X_train, y_train, sample_weight=weights)
                self._model_fit_count = len(r)

            vol_hat = float(self._model.predict(X_pred)[0])
            vol_hat = max(vol_hat, 1e-8)
            self._cache_end = end_ts
            self._cache_len = len(r)
            self._cache_vol = vol_hat
            return vol_hat

        except Exception:
            self.diag["non_converge"] += 1
            if pd.notna(end_ts):
                self.diag["non_converge_ts"].append(end_ts)
            var_hat = float(r.pow(2).tail(self.m_lookback).mean())
            return float(np.sqrt(max(var_hat, 1e-18)) * np.sqrt(self.ann))

    def estimate(self, t, returns=None):
        if returns is not None:
            r = pd.Series(returns).dropna()
            if len(r) and r.index[-1] == t:
                return self.estimate_window(r)
        return np.nan
