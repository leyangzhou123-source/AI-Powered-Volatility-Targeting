# src/estimators/har_rv.py
"""HAR-RV volatility estimator compatible with engine calling estimate(t, w)."""

import numpy as np
import pandas as pd

from src.estimators.base import Estimator


class HARRV(Estimator):
    """
    HAR-RV style estimator on realized variance proxies.

    Engine compatibility:
      - Many engines call: estimate(t, w) -> float
        where t is a Timestamp, and w is a rolling window of data.
      - This class supports:
          estimate(t, w)  (preferred in your engine)
          estimate(df)    (fallback, returns Series)

    What we do in estimate(t, w):
      - Extract returns series from w (best-effort)
      - Build RV features (rv_d, rv_w, rv_m) from r^2
      - Fit ridge regression on the window (or use fixed betas if too short)
      - Predict variance for "current step" and convert to annualized vol (float)
    """

    def __init__(self, params=None):
        super().__init__(params)

        self.returns_col = str(self.params.get("returns_col", "asset_returns"))

        self.w_lookback = int(self.params.get("w_lookback", 5))
        self.m_lookback = int(self.params.get("m_lookback", 22))

        self.ridge_lambda = float(self.params.get("ridge_lambda", 1e-6))
        self.min_obs = int(self.params.get("min_obs", 120))

        # allow either ann_factor or vol_ann
        self.ann = float(self.params.get("ann_factor", self.params.get("vol_ann", 252.0)))

    # -------------------------
    # internal helpers
    # -------------------------
    def _extract_returns(self, w) -> pd.Series:
        """
        Best-effort extraction of returns from w.
        w might be:
          - pd.DataFrame with a returns column
          - pd.Series of returns
          - np.ndarray of returns
          - dict-like
        """
        if isinstance(w, pd.Series):
            return w.astype(float)

        if isinstance(w, pd.DataFrame):
            if self.returns_col in w.columns:
                return w[self.returns_col].astype(float)
            # common fallbacks
            for c in ["asset_returns", "returns_clean", "returns_raw", "returns", "ret"]:
                if c in w.columns:
                    return w[c].astype(float)
            # last resort: if single column, use it
            if w.shape[1] == 1:
                return w.iloc[:, 0].astype(float)
            raise KeyError(f"HARRV: cannot find returns column in window df. cols={list(w.columns)}")

        if isinstance(w, np.ndarray):
            return pd.Series(w.astype(float))

        # dict-like
        if hasattr(w, "get"):
            for c in [self.returns_col, "asset_returns", "returns_clean", "returns_raw", "returns", "ret"]:
                v = w.get(c, None)
                if v is not None:
                    return pd.Series(np.asarray(v, dtype=float))

        raise TypeError(f"HARRV: unsupported window type {type(w)}")

    def _ridge_beta(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ridge regression with intercept (intercept not penalized)."""
        X1 = np.column_stack([np.ones(len(X)), X])
        p = X1.shape[1]
        I = np.eye(p)
        I[0, 0] = 0.0
        A = X1.T @ X1 + self.ridge_lambda * I
        b = X1.T @ y
        return np.linalg.solve(A, b)

    def estimate_window(self, r: pd.Series) -> float:
        """
        Given a returns window r (past returns only), fit HAR on r^2 inside window
        and forecast next-step variance using last available features.
        Returns annualized volatility (float).
        """
        r = r.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < max(self.m_lookback + 2, 10):
            return float("nan")

        rv = r.pow(2)

        # Build features aligned within window:
        # Use info up to time i-1 to predict rv at time i
        rv_d = rv
        rv_w = rv.rolling(self.w_lookback).mean()
        rv_m = rv.rolling(self.m_lookback).mean()

        X = pd.concat([rv_d.rename("rv_d"), rv_w.rename("rv_w"), rv_m.rename("rv_m")], axis=1).shift(1)
        y = rv.rename("y")
        m = pd.concat([X, y], axis=1).dropna()

        if len(m) < self.min_obs:
            # not enough to fit regression robustly; fallback to realized vol proxy
            var_hat = float(rv.tail(self.m_lookback).mean())
            var_hat = max(var_hat, 1e-18)
            return float(np.sqrt(var_hat) * np.sqrt(self.ann))

        Xn = m[["rv_d", "rv_w", "rv_m"]].to_numpy()
        yn = m["y"].to_numpy()

        beta = self._ridge_beta(Xn, yn)

        # Forecast using the last row's shifted features (which are based on the most recent past)
        rv_d_next = float(rv.iloc[-1])
        rv_w_next = float(rv.rolling(self.w_lookback).mean().iloc[-1])
        rv_m_next = float(rv.rolling(self.m_lookback).mean().iloc[-1])

        if (not np.isfinite(rv_d_next)) or (not np.isfinite(rv_w_next)) or (not np.isfinite(rv_m_next)):
            return float("nan")

        x1 = np.array([1.0, rv_d_next, rv_w_next, rv_m_next], dtype=float)
        var_hat = float(x1 @ beta)

        if not np.isfinite(var_hat):
            return float("nan")
        var_hat = max(var_hat, 1e-18)

        return float(np.sqrt(var_hat) * np.sqrt(self.ann))

    # -------------------------
    # public API: engine calls this
    # -------------------------
    def estimate(self, t, w, *args, **kwargs):
        """
        Primary entry for your engine: estimate(t, w) -> float annualized vol.

        t is unused (timestamp), w is the rolling window.
        """
        # allow vol_ann override if engine passes numeric args
        if len(args) >= 1 and isinstance(args[0], (int, float, np.integer, np.floating)) and np.isfinite(args[0]):
            self.ann = float(args[0])
        if "vol_ann" in kwargs and isinstance(kwargs["vol_ann"], (int, float)) and np.isfinite(kwargs["vol_ann"]):
            self.ann = float(kwargs["vol_ann"])

        r = self._extract_returns(w)
        return self._har_forecast_from_returns(r)

    # Optional: if some other code calls estimate(df) expecting a Series
    def estimate_series(self, df: pd.DataFrame) -> pd.Series:
        r = self._extract_returns(df)
        out = pd.Series(index=df.index, dtype=float)
        # expanding / rolling series version if you ever need it
        for i in range(len(r)):
            out.iloc[i] = self._har_forecast_from_returns(r.iloc[: i + 1])
        return out