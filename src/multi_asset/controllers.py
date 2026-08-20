"""Portfolio controllers that convert covariance estimates into asset weights."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize(weights: pd.Series, gross: float = 1.0, long_only: bool = True) -> pd.Series:
    w = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if long_only:
        w = w.clip(lower=0.0)
    total = float(w.abs().sum() if not long_only else w.sum())
    if total <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    return w * (gross / total)


def _portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    arr = weights.reindex(cov.columns).fillna(0.0).values
    return float(np.sqrt(max(arr @ cov.values @ arr, 0.0)))


def _cap_single_name(weights: pd.Series, max_weight: float) -> pd.Series:
    w = weights.clip(upper=max_weight)
    residual = 1.0 - float(w.sum())
    for _ in range(10):
        if residual <= 1e-10:
            break
        room = (max_weight - w).clip(lower=0.0)
        if room.sum() <= 0:
            break
        add = room / room.sum() * residual
        w = (w + add).clip(upper=max_weight)
        residual = 1.0 - float(w.sum())
    return _normalize(w)


class BasePortfolioController:
    name = "base_portfolio_controller"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})
        self.max_weight = float(self.params.get("max_weight", 0.45))
        self.long_only = bool(self.params.get("long_only", True))

    def raw_weights(self, cov: pd.DataFrame, returns_window: pd.DataFrame, date: pd.Timestamp, prev_weights: pd.Series) -> pd.Series:
        raise NotImplementedError

    def compute_weights(self, target_vol: float, cov: pd.DataFrame, returns_window: pd.DataFrame, date: pd.Timestamp, prev_weights: pd.Series) -> pd.Series:
        raw = self.raw_weights(cov, returns_window, date, prev_weights)
        raw = _cap_single_name(_normalize(raw, long_only=self.long_only), self.max_weight)
        vol = _portfolio_vol(raw, cov)
        scale = np.clip(target_vol / max(vol, 1e-8), 0.0, float(self.params.get("max_gross", 1.5)))
        if bool(self.params.get("vol_scale", True)):
            raw = raw * scale
        return raw


class EqualWeightController(BasePortfolioController):
    name = "equal_weight"

    def raw_weights(self, cov, returns_window, date, prev_weights):
        return pd.Series(1.0 / len(cov.columns), index=cov.columns)


class BuyAndHoldController(EqualWeightController):
    name = "buy_and_hold"

    def compute_weights(self, target_vol, cov, returns_window, date, prev_weights):
        if prev_weights is not None and float(prev_weights.abs().sum()) > 0:
            return prev_weights.reindex(cov.columns).fillna(0.0)
        return super().compute_weights(target_vol, cov, returns_window, date, prev_weights)


class InverseVolController(BasePortfolioController):
    name = "inverse_vol"

    def raw_weights(self, cov, returns_window, date, prev_weights):
        vol = pd.Series(np.sqrt(np.maximum(np.diag(cov.values), 1e-10)), index=cov.columns)
        return 1.0 / vol


class MinimumVarianceController(BasePortfolioController):
    name = "minimum_variance"

    def raw_weights(self, cov, returns_window, date, prev_weights):
        inv = np.linalg.pinv(cov.values)
        ones = np.ones(len(cov))
        w = inv @ ones
        return pd.Series(w, index=cov.columns)


class VolCappedMinimumVarianceController(MinimumVarianceController):
    name = "vol_capped_min_variance"

    def raw_weights(self, cov, returns_window, date, prev_weights):
        w = super().raw_weights(cov, returns_window, date, prev_weights)
        asset_vol = pd.Series(np.sqrt(np.maximum(np.diag(cov.values), 1e-10)), index=cov.columns)
        cap = float(self.params.get("asset_vol_cap", 0.30))
        w = w.mask(asset_vol > cap, w * cap / asset_vol)
        return w


class EqualRiskContributionController(BasePortfolioController):
    name = "equal_risk_contribution"

    def raw_weights(self, cov, returns_window, date, prev_weights):
        w = _normalize(pd.Series(1.0 / np.sqrt(np.maximum(np.diag(cov.values), 1e-10)), index=cov.columns))
        for _ in range(int(self.params.get("iterations", 80))):
            marginal = cov.values @ w.values
            risk_contrib = w.values * marginal
            target = np.mean(risk_contrib)
            adjust = target / np.maximum(risk_contrib, 1e-10)
            w = _normalize(pd.Series(w.values * np.sqrt(adjust), index=w.index))
        return w


class DiversifiedRiskParityController(EqualRiskContributionController):
    name = "diversified_risk_parity"

    def raw_weights(self, cov, returns_window, date, prev_weights):
        erc = super().raw_weights(cov, returns_window, date, prev_weights)
        corr_penalty = cov.div(np.sqrt(np.outer(np.diag(cov), np.diag(cov))), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        crowding = corr_penalty.abs().mean(axis=1)
        return erc / np.maximum(crowding, 0.25)


class MomentumTiltController(InverseVolController):
    name = "momentum_tilt"

    def raw_weights(self, cov, returns_window, date, prev_weights):
        base = super().raw_weights(cov, returns_window, date, prev_weights)
        lookback = int(self.params.get("momentum_lookback", 63))
        mom = returns_window.tail(lookback).sum(axis=0)
        tilt = (mom.rank(pct=True) - 0.5) * float(self.params.get("tilt_strength", 0.8)) + 1.0
        return base * tilt.clip(lower=0.25)


class MeanVarianceController(BasePortfolioController):
    name = "mean_variance"

    def raw_weights(self, cov, returns_window, date, prev_weights):
        lookback = int(self.params.get("mean_lookback", 126))
        mu = returns_window.tail(lookback).mean(axis=0).values * 252
        inv = np.linalg.pinv(cov.values)
        risk_aversion = float(self.params.get("risk_aversion", 6.0))
        return pd.Series(inv @ mu / risk_aversion, index=cov.columns)


class RegimeAwareRiskBudgetController(EqualRiskContributionController):
    name = "regime_aware_risk_budget"

    def raw_weights(self, cov, returns_window, date, prev_weights):
        base = super().raw_weights(cov, returns_window, date, prev_weights)
        port_proxy = returns_window.mean(axis=1)
        vol21 = float(port_proxy.tail(21).std() * np.sqrt(252))
        defensive = [c for c in base.index if c.upper() in {"IEF", "TLT", "GLD", "UUP"}]
        if vol21 > float(self.params.get("high_vol_threshold", 0.18)) and defensive:
            base.loc[defensive] *= float(self.params.get("defensive_boost", 1.6))
        return base


class DrawdownBrakePortfolioController(InverseVolController):
    name = "drawdown_brake_portfolio"

    def compute_weights(self, target_vol, cov, returns_window, date, prev_weights):
        w = super().compute_weights(target_vol, cov, returns_window, date, prev_weights)
        proxy = returns_window.mean(axis=1)
        equity = np.exp(proxy.cumsum())
        dd = float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1.0)
        brake = 1.0
        if dd < -float(self.params.get("drawdown_trigger", 0.08)):
            brake = float(self.params.get("brake_scale", 0.55))
        return w * brake


class HysteresisPortfolioController(InverseVolController):
    name = "hysteresis_portfolio"

    def compute_weights(self, target_vol, cov, returns_window, date, prev_weights):
        candidate = super().compute_weights(target_vol, cov, returns_window, date, prev_weights)
        if prev_weights is None or float(prev_weights.abs().sum()) <= 0:
            return candidate
        band = float(self.params.get("rebalance_band", 0.05))
        diff = float((candidate - prev_weights.reindex(candidate.index).fillna(0.0)).abs().sum())
        return prev_weights.reindex(candidate.index).fillna(0.0) if diff < band else candidate


CONTROLLER_REGISTRY = {
    cls.name: cls
    for cls in (
        EqualWeightController,
        BuyAndHoldController,
        InverseVolController,
        MinimumVarianceController,
        VolCappedMinimumVarianceController,
        EqualRiskContributionController,
        DiversifiedRiskParityController,
        MomentumTiltController,
        MeanVarianceController,
        RegimeAwareRiskBudgetController,
        DrawdownBrakePortfolioController,
        HysteresisPortfolioController,
    )
}
