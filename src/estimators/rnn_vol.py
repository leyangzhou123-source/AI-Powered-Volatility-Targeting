"""RNN volatility estimator (SimpleRNN with sklearn MLP fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimators.base import Estimator


class RNNVolatility(Estimator):
    def __init__(self, params=None):
        super().__init__(params)

        self.lookback = int(self.params.get("lookback", 252))
        self.seq_len = int(self.params.get("seq_len", 21))
        self.vol_ann = int(self.params.get("vol_ann", 252))
        self.min_obs = int(self.params.get("min_obs", 120))
        self.fallback = str(self.params.get("fallback", "rv")).lower()
        self.backend = str(self.params.get("backend", "sklearn_mlp")).lower()
        self.retrain_freq = int(self.params.get("retrain_freq", 21))

        self._cache_end = None
        self._cache_len = None
        self._cache_vol = None
        self._model = None
        self._backend_name = None
        self._last_fit_end = None
        self._last_fit_len = 0

        self.diag = {
            "fallback_rv": 0,
            "fallback_rv_ts": [],
            "non_converge": 0,
            "non_converge_ts": [],
            "warmup_insufficient": 0,
            "warmup_insufficient_ts": [],
            "backend": "unknown",
        }

    @staticmethod
    def _rv_fallback(r: pd.Series, vol_ann: int) -> float:
        r = pd.Series(r).dropna().astype(float)
        if len(r) < 2:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(vol_ann))

    def _build_sequences(self, r: pd.Series):
        r = pd.Series(r).dropna().astype(float)
        rv = r.rolling(window=21, min_periods=21).std(ddof=1) * np.sqrt(self.vol_ann)

        feats = pd.DataFrame({
            "ret": r,
            "sq": r**2,
            "abs": r.abs(),
            "rv_5": r.rolling(5).std(ddof=1) * np.sqrt(self.vol_ann),
            "rv_21": r.rolling(21).std(ddof=1) * np.sqrt(self.vol_ann),
        }).dropna()

        y = rv.shift(-1).reindex(feats.index)
        valid = y.notna()
        feats = feats.loc[valid]
        y = y.loc[valid]

        arr_x = feats.to_numpy(dtype=float)
        arr_y = y.to_numpy(dtype=float)
        if len(arr_x) < max(self.min_obs, self.seq_len + 5):
            return None, None, None

        X_seq = []
        Y_seq = []
        for i in range(self.seq_len, len(arr_x)):
            X_seq.append(arr_x[i - self.seq_len : i])
            Y_seq.append(arr_y[i])

        X_seq = np.asarray(X_seq, dtype=float)
        Y_seq = np.asarray(Y_seq, dtype=float)
        x_last = X_seq[-1:]
        if len(X_seq) < 20:
            return None, None, None
        return X_seq, Y_seq, x_last

    def _fit_model(self, X_seq: np.ndarray, y_seq: np.ndarray):
        # Try TensorFlow/Keras SimpleRNN first.
        if self.backend in ("auto", "keras", "tensorflow", "tf"):
            try:
                import tensorflow as tf

                tf.random.set_seed(int(self.params.get("random_state", 42)))
                model = tf.keras.Sequential([
                    tf.keras.layers.SimpleRNN(int(self.params.get("hidden_units", 16)), input_shape=(X_seq.shape[1], X_seq.shape[2])),
                    tf.keras.layers.Dense(1),
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(self.params.get("lr", 1e-3))), loss="mse")
                model.fit(
                    X_seq,
                    y_seq,
                    epochs=int(self.params.get("epochs", 20)),
                    batch_size=int(self.params.get("batch_size", 32)),
                    verbose=0,
                )
                self.diag["backend"] = "keras_simplernn"
                self._backend_name = "keras_simplernn"
                return model
            except Exception:
                if self.backend in ("keras", "tensorflow", "tf"):
                    raise

        # Fallback: MLP on flattened sequence.
        from sklearn.neural_network import MLPRegressor

        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        model = MLPRegressor(
            hidden_layer_sizes=(int(self.params.get("hidden_units", 32)),),
            activation="relu",
            solver="adam",
            max_iter=int(self.params.get("max_iter", 300)),
            random_state=int(self.params.get("random_state", 42)),
        )
        model.fit(X_flat, y_seq)
        self.diag["backend"] = "sklearn_mlp"
        self._backend_name = "sklearn_mlp"
        return model

    def _predict_model(self, x_last: np.ndarray) -> float:
        if self._model is None:
            return np.nan
        if self._backend_name == "keras_simplernn":
            pred = self._model.predict(x_last, verbose=0).reshape(-1)
            return float(pred[0])
        x_last_flat = x_last.reshape(x_last.shape[0], -1)
        return float(self._model.predict(x_last_flat)[0])

    def _needs_retrain(self, end_ts: pd.Timestamp, n: int) -> bool:
        if self._model is None:
            return True
        if self.retrain_freq <= 0:
            return False
        if self._last_fit_end == end_ts:
            return False
        return (n - self._last_fit_len) >= self.retrain_freq

    def estimate_window(self, window_returns: pd.Series) -> float:
        r = pd.Series(window_returns).dropna().astype(float)
        if len(r) == 0:
            return np.nan

        end_ts = pd.Timestamp(r.index[-1])
        n = len(r)
        if self._cache_end == end_ts and self._cache_len == n and self._cache_vol is not None:
            return float(self._cache_vol)

        if len(r) < max(self.lookback, self.min_obs):
            self.diag["warmup_insufficient"] += 1
            self.diag["warmup_insufficient_ts"].append(end_ts)
            v = self._rv_fallback(r, self.vol_ann)
            self.diag["fallback_rv"] += 1
            self.diag["fallback_rv_ts"].append(end_ts)
        else:
            r_win = r.iloc[-self.lookback:]
            X, y, x_last = self._build_sequences(r_win)
            if X is None:
                self.diag["warmup_insufficient"] += 1
                self.diag["warmup_insufficient_ts"].append(end_ts)
                v = self._rv_fallback(r_win, self.vol_ann)
                self.diag["fallback_rv"] += 1
                self.diag["fallback_rv_ts"].append(end_ts)
            else:
                try:
                    if self._needs_retrain(end_ts, n):
                        self._model = self._fit_model(X, y)
                        self._last_fit_end = end_ts
                        self._last_fit_len = n
                    v = self._predict_model(x_last)
                except Exception:
                    self.diag["non_converge"] += 1
                    self.diag["non_converge_ts"].append(end_ts)
                    v = self._rv_fallback(r_win, self.vol_ann)
                    self.diag["fallback_rv"] += 1
                    self.diag["fallback_rv_ts"].append(end_ts)

        if (not np.isfinite(v)) or (v <= 0):
            v = self._rv_fallback(r, self.vol_ann)

        self._cache_end = end_ts
        self._cache_len = n
        self._cache_vol = float(v) if np.isfinite(v) else np.nan
        return self._cache_vol

    def estimate(self, t, returns=None):
        if returns is None:
            return np.nan
        r = pd.Series(returns).dropna()
        if len(r) == 0:
            return np.nan
        return self.estimate_window(r)
