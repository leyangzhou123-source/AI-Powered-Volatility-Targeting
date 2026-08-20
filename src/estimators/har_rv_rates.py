"""
HAR-RV-Rates Volatility Estimator
===================================
Extends the classic Corsi (2009) HAR-RV model with yield-curve slope features
from Master_Dataset.parquet.

Theory
------
The Heterogeneous Autoregression of Realized Variance (HAR-RV) model captures
the multi-scale memory structure of volatility using three overlapping windows:

    RV_t = alpha + beta_d * RV_{t-1}
                 + beta_w * RV_{t-1:t-5}^(w)
                 + beta_m * RV_{t-1:t-22}^(m)
                 + gamma * Z_{t-1}
                 + epsilon_t

where Z is a vector of macro rate-slope factors.

Why slopes?
  * An inverted yield curve (t10y2y < 0) is a leading indicator of recession
    and historically precedes equity vol spikes by 6–18 months.
  * The 3m-10yr spread (t10y3m) captures near-term Fed policy expectations;
    rapid compression signals that markets price a policy pivot, which often
    coincides with elevated equity uncertainty.
  * Slope *momentum* (first difference) captures the speed of curve movement,
    which can predict vol regime transitions.

Estimation
----------
  * Works in log-variance space (log RV) to reduce skewness and force positivity.
  * Ridge regression (closed-form, no iterative solver) — extremely fast.
  * Supports fallback to plain realized vol proxy if data is insufficient.
  * Fully rolling-window compatible.

Rolling-window note
-------------------
  * The engine passes a window of `returns` (plain pd.Series).
  * Rate/VIX columns are loaded from Master_Dataset at init and aligned by date.

Engine interface
----------------
    estimate_window(window_returns: pd.Series) -> float   ← engine calls this
    estimate(t, returns)                                  ← fallback
"""

import numpy as np
import pandas as pd
from src.estimators.base import Estimator
from src.env import Env


class HARRVRates(Estimator):
    """
    HAR-RV model augmented with yield-curve slope features.

    YAML params
    -----------
    w_lookback   : int   — weekly RV window (default 5)
    m_lookback   : int   — monthly RV window (default 22)
    q_lookback   : int   — quarterly RV window (default 63)
    ridge_lambda : float — ridge regularisation (default 1e-4)
    min_obs      : int   — minimum rows after NA-drop before falling back (default 80)
    vol_ann      : int   — annualisation factor (default 252)
    log_space    : bool  — fit in log-variance space (default True)
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.w_lookback   = int(self.params.get("w_lookback",   5))
        self.m_lookback   = int(self.params.get("m_lookback",  22))
        self.q_lookback   = int(self.params.get("q_lookback",  63))
        self.ridge_lambda = float(self.params.get("ridge_lambda", 1e-4))
        self.min_obs      = int(self.params.get("min_obs",     80))
        self.ann          = float(self.params.get("vol_ann", 252.0))
        self.log_space    = bool(self.params.get("log_space",  True))

        # Diagnostics (same structure as rest of estimators)
        self.diag = {
            "fallback_rv":            0,
            "fallback_rv_ts":         [],
            "non_converge":           0,
            "non_converge_ts":        [],
            "warmup_insufficient":    0,
            "warmup_insufficient_ts": [],
        }

        # Load Master_Dataset once
        self._master_path = Env.path("processed") / "Master_Dataset.parquet"
        self._master: pd.DataFrame | None = None
        self._load_master()

    # ------------------------------------------------------------------ #
    # Data helpers                                                        #
    # ------------------------------------------------------------------ #
    def _load_master(self) -> None:
        if not self._master_path.exists():
            print(f"⚠️  HARRVRates: Master_Dataset not found at {self._master_path}. "
                  "Rate slope features will be zeroed out.")
            self._master = None
            return
        df = pd.read_parquet(self._master_path)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        self._master = df.sort_index()

    @staticmethod
    def _align(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
        """Forward-fill auxiliary series onto the dates of the current window."""
        dates_norm = pd.to_datetime(dates).tz_localize(None).normalize()
        s = series.copy()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        out = s.reindex(dates_norm).ffill().bfill()
        out.index = dates
        return out

    def _get_col(self, col: str, dates: pd.DatetimeIndex,
                 fill: float = 0.0) -> pd.Series:
        if self._master is None or col not in self._master.columns:
            return pd.Series(fill, index=dates)
        return self._align(self._master[col].astype(float), dates)

    # ------------------------------------------------------------------ #
    # Ridge regression helper                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ridge_solve(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        """
        Closed-form Ridge with intercept (intercept is NOT penalised).
        Augments X with a ones column, zeros out the intercept penalty.
        """
        X1 = np.column_stack([np.ones(len(X)), X])
        p  = X1.shape[1]
        I  = np.eye(p)
        I[0, 0] = 0.0               # do not penalise intercept
        A  = X1.T @ X1 + lam * I
        b  = X1.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(X1, y, rcond=None)[0]

    # ------------------------------------------------------------------ #
    # Feature construction                                                #
    # ------------------------------------------------------------------ #
    def _build_features(self, r: pd.Series) -> pd.DataFrame:
        """
        Build HAR + rate-slope feature matrix.
        All inputs are shifted by 1 to prevent look-ahead bias.
        """
        dates = r.index

        # Use returns_clean if available (removes roll-day spikes)
        r_clean = self._get_col("returns_clean", dates, fill=np.nan)
        r_clean = r_clean.where(r_clean.notna(), r)

        rv   = r_clean.pow(2)
        rv_d = rv
        rv_w = rv.rolling(self.w_lookback).mean()
        rv_m = rv.rolling(self.m_lookback).mean()
        rv_q = rv.rolling(self.q_lookback).mean()

        # Rate slope features
        slope_2y10y   = self._get_col("t10y2y",  dates, fill=np.nan)
        slope_3m10y   = self._get_col("t10y3m",  dates, fill=np.nan)
        d_slope_2y10y = slope_2y10y.diff()
        d_slope_3m10y = slope_3m10y.diff()
        inv_2y10y     = (slope_2y10y < 0).astype(float)   # inversion dummy

        # VIX (optional enrichment — forward-fill from master)
        vix_raw  = self._get_col("vix_close", dates, fill=np.nan)
        vix_d    = (vix_raw / 100.0) ** 2 / self.ann      # daily variance scale

        features = pd.concat([
            rv_d.rename("rv_d"),
            rv_w.rename("rv_w"),
            rv_m.rename("rv_m"),
            rv_q.rename("rv_q"),
            slope_2y10y.rename("slope_2y10y"),
            slope_3m10y.rename("slope_3m10y"),
            d_slope_2y10y.rename("d_slope_2y10y"),
            d_slope_3m10y.rename("d_slope_3m10y"),
            inv_2y10y.rename("inv_2y10y"),
            vix_d.rename("vix_d"),
        ], axis=1).shift(1)    # ← all features shifted 1 step forward

        return features

    # ------------------------------------------------------------------ #
    # Core estimation                                                     #
    # ------------------------------------------------------------------ #
    def _rv_fallback(self, r: pd.Series) -> float:
        v = float(r.pow(2).tail(self.m_lookback).mean())
        return float(np.sqrt(max(v, 1e-18)) * np.sqrt(self.ann))

    def estimate_window(self, window_returns: pd.Series) -> float:
        """Called by the engine at each rolling step."""
        r      = pd.Series(window_returns).dropna().astype(float)
        end_ts = pd.Timestamp(r.index[-1]) if len(r) else pd.NaT

        # Warm-up guard
        if len(r) < max(self.q_lookback + 2, self.min_obs):
            self.diag["warmup_insufficient"] += 1
            if pd.notna(end_ts):
                self.diag["warmup_insufficient_ts"].append(end_ts)
            return float("nan")

        X_df = self._build_features(r)

        # Target: log-RV or raw RV
        rv   = r.pow(2)
        y_s  = np.log(rv + 1e-8) if self.log_space else rv

        m    = pd.concat([X_df, y_s.rename("y")], axis=1).dropna()

        if len(m) < self.min_obs:
            self.diag["fallback_rv"] += 1
            if pd.notna(end_ts):
                self.diag["fallback_rv_ts"].append(end_ts)
            return self._rv_fallback(r)

        feat_cols = [c for c in m.columns if c != "y"]
        Xn        = m[feat_cols].to_numpy()
        yn        = m["y"].to_numpy()

        # Separate train (all but last row) and prediction (last row)
        X_train = Xn[:-1]
        y_train = yn[:-1]
        x_pred  = Xn[-1:]

        try:
            beta    = self._ridge_solve(X_train, y_train, self.ridge_lambda)
            x_pred1 = np.column_stack([np.ones(1), x_pred])
            y_hat   = float(x_pred1 @ beta)

            if not np.isfinite(y_hat):
                raise ValueError("non-finite prediction")

            # Convert back to annualised vol
            if self.log_space:
                var_hat = np.exp(y_hat)
            else:
                var_hat = max(y_hat, 1e-18)

            return float(np.sqrt(max(var_hat, 1e-18)) * np.sqrt(self.ann))

        except Exception:
            self.diag["non_converge"] += 1
            if pd.notna(end_ts):
                self.diag["non_converge_ts"].append(end_ts)
            return self._rv_fallback(r)

    def estimate(self, t, returns=None):
        if returns is not None:
            r = pd.Series(returns).dropna()
            if len(r) and r.index[-1] == t:
                return self.estimate_window(r)
        return np.nan

    # Optional: full-series estimate for offline use
    def fit(self, returns: pd.Series):
        """Not needed for rolling-window engine but satisfies base class."""
        return self
