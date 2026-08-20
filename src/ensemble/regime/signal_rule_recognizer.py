"""Direct signal-rule regime recognizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ensemble.regime.base import BaseRegimeRecognizer
from src.ensemble.regime.regime_labels import NORMAL, STRESS, TRANSITION
from src.ensemble.signals import (
    compute_realized_vol,
    iv_rv_spread,
    iv_rv_spread_bucket,
    liquidity_stress_proxy,
    rolling_correlation_spike,
    volatility_level_bucket,
    volatility_trend,
    vix_level_bucket,
    load_vix_series,
)
from src.ensemble.utils import infer_returns_series


class SignalRuleRecognizer(BaseRegimeRecognizer):
    def __init__(self, params=None):
        super().__init__(params)
        self.signal_cfg = dict(self.params.get("signals", {}))
        self.rules = list(self.params.get("rules", []))

    def fit(self, df: pd.DataFrame):
        self._fitted = True
        return self

    @staticmethod
    def _parse_condition(token: str) -> tuple[str, str]:
        left, right = str(token).split("=", 1)
        return left.strip(), right.strip()

    def _build_signal_table(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = infer_returns_series(df, returns_col="returns")
        out = pd.DataFrame(index=pd.to_datetime(df.index))

        rv_cfg = self.signal_cfg.get("volatility_level", {})
        if rv_cfg.get("enabled", True):
            rv = compute_realized_vol(
                returns,
                lookback=int(rv_cfg.get("lookback", 21)),
                ann_factor=float(rv_cfg.get("ann_factor", 252.0)),
            )
            out["signal_volatility_level"] = volatility_level_bucket(
                rv,
                low_quantile=float(rv_cfg.get("low_quantile", 0.3)),
                high_quantile=float(rv_cfg.get("high_quantile", 0.7)),
            ).reindex(out.index)
            out["signal_realized_vol"] = rv.reindex(out.index)

        vt_cfg = self.signal_cfg.get("volatility_trend", {})
        if vt_cfg.get("enabled", True):
            base_rv = out.get("signal_realized_vol")
            if base_rv is None:
                base_rv = compute_realized_vol(returns)
            out["signal_volatility_trend"] = volatility_trend(
                pd.Series(base_rv),
                lookback=int(vt_cfg.get("lookback", 5)),
                threshold=float(vt_cfg.get("threshold", 0.0)),
            ).reindex(out.index)

        vix_cfg = self.signal_cfg.get("vix_level", {})
        if vix_cfg.get("enabled", False):
            vix_path = Path(vix_cfg.get("path", "data/processed/VIX_Daily_Processed.parquet"))
            if not vix_path.is_absolute():
                vix_path = Path.cwd() / vix_path
            vix = load_vix_series(vix_path)
            out["signal_vix"] = vix.reindex(out.index)
            out["signal_vix_level"] = vix_level_bucket(
                out["signal_vix"],
                low_quantile=float(vix_cfg.get("low_quantile", 0.3)),
                high_quantile=float(vix_cfg.get("high_quantile", 0.7)),
            )

        spread_cfg = self.signal_cfg.get("iv_rv_spread", {})
        if spread_cfg.get("enabled", False):
            if "signal_vix" not in out.columns:
                vix = load_vix_series(Path.cwd() / "data/processed/VIX_Daily_Processed.parquet")
                out["signal_vix"] = vix.reindex(out.index)
            base_rv = out.get("signal_realized_vol")
            if base_rv is None:
                base_rv = compute_realized_vol(returns, lookback=int(spread_cfg.get("rv_lookback", 21)))
            spread = iv_rv_spread(pd.Series(out["signal_vix"]), pd.Series(base_rv))
            out["signal_iv_rv_spread"] = spread.reindex(out.index)
            out["signal_iv_rv_spread_level"] = iv_rv_spread_bucket(
                out["signal_iv_rv_spread"],
                threshold_quantile=float(spread_cfg.get("threshold_quantile", 0.7)),
            )

        corr_cfg = self.signal_cfg.get("correlation_spike", {})
        if corr_cfg.get("enabled", False):
            out["signal_correlation_spike"] = rolling_correlation_spike(
                df,
                lookback=int(corr_cfg.get("lookback", 21)),
                threshold=float(corr_cfg.get("threshold", 0.8)),
                columns=corr_cfg.get("columns"),
            ).reindex(out.index)

        liq_cfg = self.signal_cfg.get("liquidity_stress", {})
        if liq_cfg.get("enabled", False):
            out["signal_liquidity_stress"] = liquidity_stress_proxy(
                df,
                columns=liq_cfg.get("columns"),
                threshold_quantile=float(liq_cfg.get("threshold_quantile", 0.8)),
            ).reindex(out.index)

        return out

    def _match_rule(self, row: pd.Series, rule_if: list[str]) -> bool:
        for cond in rule_if:
            k, v = self._parse_condition(cond)
            signal_key = f"signal_{k}"
            if signal_key not in row.index:
                return False
            if str(row[signal_key]) != v:
                return False
        return True

    def _apply_rules(self, signal_table: pd.DataFrame) -> pd.Series:
        label = pd.Series(NORMAL, index=signal_table.index, dtype="object")

        # Fallback defaults if no explicit rules configured.
        if not self.rules:
            if "signal_volatility_level" in signal_table.columns and "signal_vix_level" in signal_table.columns:
                is_stress = (
                    (signal_table["signal_volatility_level"] == "high")
                    & (signal_table["signal_vix_level"] == "high")
                )
                label[is_stress.fillna(False)] = STRESS
            if "signal_volatility_trend" in signal_table.columns:
                is_transition = signal_table["signal_volatility_trend"] == "rising"
                label[is_transition.fillna(False)] = TRANSITION
            return label

        default_regime = NORMAL
        for rule in self.rules:
            if "default" in rule:
                default_regime = str(rule["default"])
                continue
            conds = list(rule.get("if", []))
            regime = str(rule.get("regime", NORMAL))
            if not conds:
                continue
            mask = signal_table.apply(lambda row: self._match_rule(row, conds), axis=1)
            label.loc[mask] = regime

        return label.fillna(default_regime)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df)

        signal_table = self._build_signal_table(df)
        label = self._apply_rules(signal_table)

        out = signal_table.copy()
        out["regime_label"] = label.reindex(out.index).fillna(NORMAL)

        # Optional soft scores from simple heuristics for downstream weighted blending.
        out["regime_prob_stress"] = np.where(out["regime_label"] == STRESS, 1.0, 0.0)
        out["regime_prob_transition"] = np.where(out["regime_label"] == TRANSITION, 1.0, 0.0)
        out["regime_prob_normal"] = np.where(
            ~(out["regime_label"].isin([STRESS, TRANSITION])),
            1.0,
            0.0,
        )

        return out
