# src/estimators/hybrid_ewma_regime.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.estimators.base import Estimator


class HybridEWMARegime(Estimator):
    """
    Hybrid EWMA + 2-state HMM estimator.

    Responsibilities moved here from engine:
        - _forward_filter_probs
        - _fit_two_state_hmm
        - _compute_divergence_and_z
        - _build_hybrid_signal_table

    Public engine-facing behavior:
        - estimate_window_components(w)
        - estimate_window(w)                      # fallback
        - build_vol_series(dates, returns, window, rebal_dates=None, config=None)

    The returned signal table contains vol_estimate, but the engine only needs
    the vol_estimate column for backtesting.
    """

    uses_hybrid_regime_vol = True

    def __init__(self, params=None):
        super().__init__(params)
        self.params = params or {}

        self.fast_half_life = float(self.params.get("fast_half_life", 5))
        self.slow_half_life = float(self.params.get("slow_half_life", 40))
        self.ann_factor = float(self.params.get("ann_factor", self.params.get("vol_ann", 252.0)))
        self.var_init_mode = str(self.params.get("var_init_mode", "window_var")).lower()
        self.min_obs = int(self.params.get("min_obs", 30))
        self.eps = float(self.params.get("eps", 1e-12))

    def _lambda_from_half_life(self, half_life: float) -> float:
        half_life = max(float(half_life), 1e-8)
        return float(0.5 ** (1.0 / half_life))

    def _initial_variance(self, r: np.ndarray) -> float:
        if len(r) == 0:
            return self.eps

        if self.var_init_mode == "last_sq":
            v0 = float(r[-1] ** 2)
        else:
            if len(r) > 1:
                v0 = float(np.nanvar(r, ddof=1))
            else:
                v0 = float(r[-1] ** 2)

        return max(v0, self.eps)

    def _ewma_vol_path(self, returns: pd.Series, half_life: float) -> pd.Series:
        x = pd.Series(returns, dtype=float).dropna()
        if len(x) == 0:
            return pd.Series(dtype=float, index=x.index)

        lam = self._lambda_from_half_life(half_life)
        r = x.values.astype(float)

        var_t = self._initial_variance(r)
        out = np.full(len(r), np.nan, dtype=float)

        for i, ret in enumerate(r):
            var_t = lam * var_t + (1.0 - lam) * float(ret ** 2)
            var_t = max(var_t, self.eps)
            out[i] = np.sqrt(var_t * self.ann_factor)

        return pd.Series(out, index=x.index, name=f"ewma_hl_{half_life}")

    def estimate_window_components(self, w: pd.Series) -> dict:
        x = pd.Series(w, dtype=float).dropna()

        if len(x) < self.min_obs:
            return {
                "sigma_fast": np.nan,
                "sigma_slow": np.nan,
                "sigma_fast_hist": pd.Series(dtype=float),
                "sigma_slow_hist": pd.Series(dtype=float),
            }

        sigma_fast_hist = self._ewma_vol_path(x, self.fast_half_life)
        sigma_slow_hist = self._ewma_vol_path(x, self.slow_half_life)

        sigma_fast = float(sigma_fast_hist.iloc[-1]) if len(sigma_fast_hist) else np.nan
        sigma_slow = float(sigma_slow_hist.iloc[-1]) if len(sigma_slow_hist) else np.nan

        return {
            "sigma_fast": sigma_fast,
            "sigma_slow": sigma_slow,
            "sigma_fast_hist": sigma_fast_hist,
            "sigma_slow_hist": sigma_slow_hist,
        }

    def estimate_components(self, w: pd.Series) -> dict:
        return self.estimate_window_components(w)

    def estimate_window(self, w: pd.Series) -> float:
        comps = self.estimate_window_components(w)
        sigma_slow = comps.get("sigma_slow", np.nan)
        return float(sigma_slow) if np.isfinite(sigma_slow) else np.nan

    def _forward_filter_probs(self, model, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        T = obs.shape[0]
        K = model.n_components

        startprob = np.asarray(model.startprob_, dtype=float)
        transmat = np.asarray(model.transmat_, dtype=float)

        framelogprob = model._compute_log_likelihood(obs)
        emission = np.exp(framelogprob)

        alpha = np.zeros((T, K), dtype=float)

        alpha[0] = startprob * emission[0]
        s0 = alpha[0].sum()
        alpha[0] = np.ones(K) / K if (s0 <= 0 or not np.isfinite(s0)) else alpha[0] / s0

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ transmat) * emission[t]
            st = alpha[t].sum()
            alpha[t] = np.ones(K) / K if (st <= 0 or not np.isfinite(st)) else alpha[t] / st

        return alpha

    def _fit_two_state_hmm(self, obs_hist: np.ndarray) -> dict:
        obs_hist = np.asarray(obs_hist, dtype=float)
        if obs_hist.ndim == 1:
            obs_hist = obs_hist.reshape(-1, 1)

        mu = float(obs_hist.mean())
        sd = float(obs_hist.std())
        sd = max(sd, 1e-6)

        x = (obs_hist - mu) / sd

        model = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=200,
            tol=1e-4,
            min_covar=1e-4,
            random_state=42,
            init_params="",
            params="stmc",
        )

        model.startprob_ = np.array([0.90, 0.10], dtype=float)
        model.transmat_ = np.array([
            [0.97, 0.03],
            [0.08, 0.92],
        ], dtype=float)
        model.means_ = np.array([[-0.5], [0.5]], dtype=float)
        model.covars_ = np.array([[1.0], [1.0]], dtype=float)

        model.fit(x)

        filtered = self._forward_filter_probs(model, x)
        p_t = filtered[-1]
        p_next = p_t @ model.transmat_

        state_means = model.means_.reshape(-1)
        order = np.argsort(state_means)
        normal_state = int(order[0])
        stress_state = int(order[1])

        trans = np.asarray(model.transmat_, dtype=float)

        return {
            "filtered_probs": p_t,
            "next_probs": p_next,
            "normal_state": normal_state,
            "stress_state": stress_state,
            "transition_p00": float(trans[normal_state, normal_state]),
            "transition_p01": float(trans[normal_state, stress_state]),
            "transition_p10": float(trans[stress_state, normal_state]),
            "transition_p11": float(trans[stress_state, stress_state]),
        }

    def _compute_divergence_and_z(
        self,
        sigma_fast_hist: pd.Series,
        sigma_slow_hist: pd.Series,
        z_window: int,
        eps: float,
    ):
        d = (
            np.log(sigma_fast_hist.clip(lower=eps))
            - np.log(sigma_slow_hist.clip(lower=eps))
        ).abs()

        min_periods = max(20, z_window // 5)
        mu = d.rolling(z_window, min_periods=min_periods).mean()
        sd = d.rolling(z_window, min_periods=min_periods).std(ddof=1).clip(lower=eps)
        z = (d - mu) / sd

        d_t = float(d.iloc[-1]) if len(d) and np.isfinite(d.iloc[-1]) else np.nan
        z_t = float(z.iloc[-1]) if len(z) and np.isfinite(z.iloc[-1]) else np.nan

        return d_t, z_t

    def _build_hybrid_signal_table(
        self,
        dates: pd.DatetimeIndex,
        returns_clean: pd.Series,
        window: int,
        rebal_dates: set | None = None,
        config: dict | None = None,
    ) -> pd.DataFrame:
        cfg_h = (config or {}).get("hybrid_signal", {})
        z_window = int(cfg_h.get("z_window", 126))
        hmm_min_obs = int(cfg_h.get("hmm_min_obs", 60))
        smooth_span = int(cfg_h.get("smooth_span", 5))
        eps = float(cfg_h.get("eps", 1e-8))
        k = float(cfg_h.get("k", 1.0))
        z0 = float(cfg_h.get("z0", 1.5))
        gamma = float(cfg_h.get("gamma", 1.0))

        n = len(dates)
        if n <= window:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

        out_idx = dates[window:]
        rows = []
        last_row = None

        for i in range(window, n):
            t = dates[i]

            if rebal_dates is not None and t not in rebal_dates and last_row is not None:
                row = dict(last_row)
                row["date"] = t
                rows.append(row)
                continue

            row = {
                "vol_estimate": np.nan,
            }

            w = returns_clean.iloc[i - window:i].dropna()
            if len(w) == 0:
                row["date"] = t
                rows.append(row)
                last_row = row
                continue

            try:
                est = self.estimate_window_components(w)
            except Exception:
                row["date"] = t
                rows.append(row)
                last_row = row
                continue

            sigma_fast = float(est.get("sigma_fast", np.nan))
            sigma_slow = float(est.get("sigma_slow", np.nan))
            sigma_fast_hist = est.get("sigma_fast_hist", None)
            sigma_slow_hist = est.get("sigma_slow_hist", None)

            if sigma_fast_hist is None or sigma_slow_hist is None:
                row["date"] = t
                rows.append(row)
                last_row = row
                continue

            sigma_fast_hist = pd.Series(sigma_fast_hist, dtype=float).dropna()
            sigma_slow_hist = pd.Series(sigma_slow_hist, dtype=float).dropna()

            n_hist = min(len(sigma_fast_hist), len(sigma_slow_hist))
            if n_hist == 0:
                row["date"] = t
                rows.append(row)
                last_row = row
                continue

            sigma_fast_hist = sigma_fast_hist.iloc[-n_hist:]
            sigma_slow_hist = sigma_slow_hist.iloc[-n_hist:]

            _, z_t = self._compute_divergence_and_z(
                sigma_fast_hist=sigma_fast_hist,
                sigma_slow_hist=sigma_slow_hist,
                z_window=z_window,
                eps=eps,
            )

            obs_series = sigma_slow_hist.copy()
            obs_series = obs_series[(np.isfinite(obs_series)) & (obs_series > 0)]

            if len(obs_series) >= hmm_min_obs:
                if smooth_span > 1:
                    obs_used = (
                        np.log(obs_series)
                        .ewm(span=smooth_span, adjust=False)
                        .mean()
                        .values.reshape(-1, 1)
                    )
                else:
                    obs_used = np.log(obs_series).values.reshape(-1, 1)

                try:
                    hmm = self._fit_two_state_hmm(obs_used)

                    p_t = hmm["filtered_probs"]
                    normal_state = hmm["normal_state"]
                    stress_state = hmm["stress_state"]

                    p_stress = float(p_t[stress_state])

                    pi01 = hmm["transition_p01"]
                    pi10 = hmm["transition_p10"]
                    denom = pi01 + pi10
                    p_bar = (pi01 / denom) if denom > eps else p_stress

                    if np.isfinite(z_t) and np.isfinite(sigma_fast) and np.isfinite(sigma_slow):
                        z0_t = z0 - gamma * (p_stress - p_bar)
                        beta = 1.0 / (1.0 + np.exp(-k * (z_t - z0_t)))
                        chosen_vol = beta * sigma_fast + (1.0 - beta) * sigma_slow
                        row["vol_estimate"] = float(chosen_vol)

                except Exception:
                    pass

            row["date"] = t
            rows.append(row)
            last_row = row

        out = pd.DataFrame(rows).set_index("date")
        out.index = pd.to_datetime(out.index)
        out = out.reindex(out_idx)
        return out

    def build_vol_series(
        self,
        dates: pd.DatetimeIndex,
        returns_clean: pd.Series,
        window: int,
        rebal_dates: set | None = None,
        config: dict | None = None,
    ) -> pd.Series:
        signal_table = self._build_hybrid_signal_table(
            dates=dates,
            returns_clean=returns_clean,
            window=window,
            rebal_dates=rebal_dates,
            config=config,
        )
        if "vol_estimate" not in signal_table.columns:
            return pd.Series(dtype=float, index=signal_table.index, name="vol_estimate")
        return signal_table["vol_estimate"].astype(float).rename("vol_estimate")