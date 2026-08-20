"""Base rule-based router for selecting estimator/controller pairs.

This implements the paper's base routing rule:

    Score_t(k) = Perf_t(k) + RegimeBias_t(k)
                 - lambda_inv * InvalidRate_t(k)
                 - lambda_exc * ExceptionRate_t(k)

The implementation keeps the existing ``Router`` name for backwards
compatibility with configs that use ``router.type: base``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from src.router.strategy_pair import StrategyPair


class Router:
    """Rule-based selector over estimator-controller pairs.

    Expected inputs are intentionally plain dictionaries so the router can be
    used from both the live backtest engine and walk-forward protocol scripts.
    ``performance_metrics`` may be either global metrics, or a mapping keyed by
    pair name. Pair-specific metrics are preferred when available.
    """

    def __init__(self, pairs: list[StrategyPair], params: dict[str, Any] | None = None):
        if not pairs:
            raise ValueError("Router requires at least one StrategyPair.")

        self.params = params or {}
        self.excluded_estimators = self._as_lower_list(self.params.get("excluded_estimators", []))
        self.excluded_controllers = self._as_lower_list(self.params.get("excluded_controllers", []))
        self.excluded_pairs_containing = self._as_lower_list(self.params.get("excluded_pairs_containing", []))
        self.pairs = self._filter_pairs(pairs)
        if not self.pairs:
            raise ValueError("Router exclusions removed every StrategyPair.")
        self.default_pair = self._resolve_default_pair(self.params.get("default_pair"))
        self.sticky_period = int(self.params.get("sticky_period", 1))
        self.min_performance_obs = int(self.params.get("min_performance_obs", 20))
        self.use_perf_weight = bool(self.params.get("use_perf_weight", True))

        self.perf_weight = float(self.params.get("perf_weight", 1.0))
        self.regime_bias_weight = float(self.params.get("regime_bias_weight", 1.0))
        self.lambda_invalid = float(self.params.get("lambda_invalid", self.params.get("lambda_inv", 2.0)))
        self.lambda_exception = float(self.params.get("lambda_exception", self.params.get("lambda_exc", 1.0)))
        self.lambda_drawdown = float(self.params.get("lambda_drawdown", 0.5))
        self.lambda_switch = float(self.params.get("lambda_switch", 0.0))

        self.regime_bias = self.params.get("regime_bias", {}) or {}
        self.estimator_regime_bias = self.params.get("estimator_regime_bias", {}) or {}
        self.controller_regime_bias = self.params.get("controller_regime_bias", {}) or {}
        self.use_heuristic_regime_bias = bool(self.params.get("use_heuristic_regime_bias", True))
        self.regime_suitability_scale = float(self.params.get("regime_suitability_scale", 0.5))
        self._load_regime_suitability(self.params.get("regime_suitability_path"))

        self._active_pair = self.default_pair
        self._active_since = 0
        self.decisions: list[dict[str, Any]] = []

    @staticmethod
    def _as_lower_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.lower()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).lower() for item in value]
        return [str(value).lower()]

    def _pair_search_text(self, pair: StrategyPair) -> tuple[str, str, str]:
        metadata = pair.metadata if isinstance(pair.metadata, dict) else {}
        estimator_text = " ".join(
            [
                pair.estimator_name,
                str(metadata.get("estimator_path", "")),
            ]
        ).lower()
        controller_text = " ".join(
            [
                pair.controller_name,
                str(metadata.get("controller_path", "")),
            ]
        ).lower()
        pair_text = " ".join([pair.name, estimator_text, controller_text]).lower()
        return pair_text, estimator_text, controller_text

    def _is_excluded_pair(self, pair: StrategyPair) -> bool:
        pair_text, estimator_text, controller_text = self._pair_search_text(pair)
        if any(token in estimator_text for token in self.excluded_estimators):
            return True
        if any(token in controller_text for token in self.excluded_controllers):
            return True
        if any(token in pair_text for token in self.excluded_pairs_containing):
            return True
        return False

    def _filter_pairs(self, pairs: list[StrategyPair]) -> list[StrategyPair]:
        if not (self.excluded_estimators or self.excluded_controllers or self.excluded_pairs_containing):
            return pairs
        return [pair for pair in pairs if not self._is_excluded_pair(pair)]

    def _resolve_default_pair(self, default_name: str | None) -> StrategyPair:
        if default_name is None:
            return self.pairs[0]

        for pair in self.pairs:
            if pair.name == default_name:
                return pair

        return self.pairs[0]

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v

    @staticmethod
    def _canonical_regime(value: Any) -> str:
        text = str(value).lower()
        if text in ("mid", "middle", "normal"):
            return "middle"
        if text in ("low", "high"):
            return text
        return text

    @staticmethod
    def _merge_nested_scores(target: dict[str, Any], source: dict[str, Any]) -> None:
        for regime, values in (source or {}).items():
            if not isinstance(values, dict):
                continue
            regime_key = Router._canonical_regime(regime)
            target.setdefault(regime_key, {})
            target[regime_key].update(values)

    def _load_regime_suitability(self, path_value: Any) -> None:
        """Load pair/estimator/controller regime priors from YAML or CSV outputs."""
        if not path_value:
            return

        path = Path(str(path_value))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"regime_suitability_path not found: {path}")

        if path.is_dir():
            self._load_regime_suitability_dir(path)
            return

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        params = raw.get("router", {}).get("params", raw) if isinstance(raw, dict) else {}
        if not isinstance(params, dict):
            return

        self._merge_nested_scores(self.regime_bias, params.get("regime_bias", {}))
        self._merge_nested_scores(self.estimator_regime_bias, params.get("estimator_regime_bias", {}))
        self._merge_nested_scores(self.controller_regime_bias, params.get("controller_regime_bias", {}))

    def _load_regime_suitability_dir(self, path: Path) -> None:
        self._load_suitability_csv(path / "pair_regime_suitability.csv", "pair", self.regime_bias)
        self._load_suitability_csv(
            path / "estimator_regime_suitability.csv",
            "estimator",
            self.estimator_regime_bias,
        )
        self._load_suitability_csv(
            path / "controller_regime_suitability.csv",
            "controller",
            self.controller_regime_bias,
        )

    def _load_suitability_csv(self, path: Path, key_col: str, target: dict[str, Any]) -> None:
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                regime = self._canonical_regime(row.get("regime", ""))
                key = row.get(key_col)
                if not regime or not key:
                    continue
                score = self._safe_float(row.get("suitability_score"), 0.0)
                target.setdefault(regime, {})[key] = score * self.regime_suitability_scale

    def _pair_performance_metrics(
        self,
        pair: StrategyPair,
        performance_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Return pair-specific metrics if provided, otherwise global metrics."""
        pair_metrics = performance_metrics.get(pair.name)
        if isinstance(pair_metrics, dict):
            return pair_metrics
        return performance_metrics

    def _pair_diagnostics(self, pair: StrategyPair, diagnostics: dict[str, Any]) -> dict[str, Any]:
        pair_diag = diagnostics.get(pair.name, {})
        return pair_diag if isinstance(pair_diag, dict) else {}

    def _performance_score(self, pair: StrategyPair, performance_metrics: dict[str, Any]) -> float:
        if not self.use_perf_weight:
            return 0.0

        pair_perf = self._pair_performance_metrics(pair, performance_metrics)
        obs = int(self._safe_float(pair_perf.get("obs", 0), 0.0))
        if obs < self.min_performance_obs:
            return 0.0

        sharpe = self._safe_float(pair_perf.get("rolling_sharpe", 0.0))
        drawdown = self._safe_float(pair_perf.get("drawdown", 0.0))
        return self.perf_weight * (sharpe - self.lambda_drawdown * max(drawdown, 0.0))

    def _lookup_regime_score(self, table: dict[str, Any], vol_regime: str, key: str) -> float | None:
        regime = self._canonical_regime(vol_regime)
        if not isinstance(table, dict):
            return None
        regime_map = table.get(regime, {})
        if isinstance(regime_map, dict) and key in regime_map:
            return self._safe_float(regime_map[key])
        return None

    def _configured_regime_bias(self, pair: StrategyPair, vol_regime: str) -> float | None:
        """Read optional config/metadata regime bias.

        Supported config shape:
            router.params.regime_bias:
              high:
                garch__regime: 0.4
              low:
                ewma__naive: 0.2
        Pair metadata can also contain the same regime-to-score mapping under
        ``metadata["regime_bias"]``.
        """
        metadata_bias = pair.metadata.get("regime_bias", {}) if isinstance(pair.metadata, dict) else {}
        regime = self._canonical_regime(vol_regime)
        if isinstance(metadata_bias, dict) and regime in metadata_bias:
            return self._safe_float(metadata_bias[regime])

        return self._lookup_regime_score(self.regime_bias, regime, pair.name)

    def _regime_bias_components(
        self,
        pair: StrategyPair,
        market_features: dict[str, Any],
    ) -> dict[str, float]:
        vol_regime = self._canonical_regime(market_features.get("vol_regime", "middle"))
        pair_bias = self._configured_regime_bias(pair, vol_regime)
        estimator_bias = self._lookup_regime_score(
            self.estimator_regime_bias,
            vol_regime,
            pair.estimator_name,
        )
        controller_bias = self._lookup_regime_score(
            self.controller_regime_bias,
            vol_regime,
            pair.controller_name,
        )

        configured_total = sum(
            value for value in (pair_bias, estimator_bias, controller_bias) if value is not None
        )

        heuristic = 0.0
        if configured_total == 0.0 and self.use_heuristic_regime_bias:
            heuristic = self._heuristic_regime_bias(pair, vol_regime)

        return {
            "pair_regime_bias": float(pair_bias or 0.0),
            "estimator_regime_bias": float(estimator_bias or 0.0),
            "controller_regime_bias": float(controller_bias or 0.0),
            "heuristic_regime_bias": float(heuristic),
            "total": float(self.regime_bias_weight * (configured_total + heuristic)),
        }

    def _regime_bias_score(self, pair: StrategyPair, market_features: dict[str, Any]) -> float:
        return self._regime_bias_components(pair, market_features)["total"]

    def _heuristic_regime_bias(self, pair: StrategyPair, vol_regime: str) -> float:
        estimator = pair.estimator_name.lower()
        controller = pair.controller_name.lower()

        bias = 0.0
        if vol_regime == "high":
            if "regime" in controller or "drawdown" in controller or "cvar" in controller:
                bias += 0.35
            if "garch" in estimator or "har" in estimator:
                bias += 0.25
        elif vol_regime == "low":
            if "naive" in controller or "constant" in controller or "hysteresis" in controller:
                bias += 0.20
            if "ewma" in estimator or "realized" in estimator:
                bias += 0.15
        elif vol_regime == "middle":
            if "voltargetclip" in controller or "variance" in controller:
                bias += 0.10

        return bias

    def _diagnostic_penalty(self, pair: StrategyPair, diagnostics: dict[str, Any]) -> float:
        pair_diag = self._pair_diagnostics(pair, diagnostics)
        invalid_rate = self._safe_float(pair_diag.get("invalid_rate", 0.0))
        exception_rate = self._safe_float(pair_diag.get("exception_rate", 0.0))
        return self.lambda_invalid * invalid_rate + self.lambda_exception * exception_rate

    def _switch_penalty(self, pair: StrategyPair) -> float:
        if self._active_pair is None or pair.name == self._active_pair.name:
            return 0.0
        return self.lambda_switch

    def _score_components(
        self,
        pair: StrategyPair,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> dict[str, float]:
        performance = self._performance_score(pair, performance_metrics)
        regime_components = self._regime_bias_components(pair, market_features)
        regime_bias = regime_components["total"]
        diagnostic_penalty = self._diagnostic_penalty(pair, diagnostics)
        switch_penalty = self._switch_penalty(pair)
        total = performance + regime_bias - diagnostic_penalty - switch_penalty
        return {
            "performance": float(performance),
            "regime_bias": float(regime_bias),
            "pair_regime_bias": float(regime_components["pair_regime_bias"]),
            "estimator_regime_bias": float(regime_components["estimator_regime_bias"]),
            "controller_regime_bias": float(regime_components["controller_regime_bias"]),
            "heuristic_regime_bias": float(regime_components["heuristic_regime_bias"]),
            "diagnostic_penalty": float(diagnostic_penalty),
            "switch_penalty": float(switch_penalty),
            "total": float(total),
        }

    def _score_pair(
        self,
        pair: StrategyPair,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> float:
        return self._score_components(pair, market_features, diagnostics, performance_metrics)["total"]

    def select(
        self,
        market_features: dict[str, Any],
        diagnostics: dict[str, Any],
        performance_metrics: dict[str, Any],
        timestamp: Any = None,
    ) -> StrategyPair:
        if self._active_pair is None:
            self._active_pair = self.default_pair

        can_switch = (self._active_since + 1) >= self.sticky_period

        best_pair = self._active_pair
        best_score = float("-inf")

        scores: dict[str, float] = {}
        score_components: dict[str, dict[str, float]] = {}
        for pair in self.pairs:
            components = self._score_components(pair, market_features, diagnostics, performance_metrics)
            s = components["total"]
            scores[pair.name] = s
            score_components[pair.name] = components
            if s > best_score:
                best_score = s
                best_pair = pair

        switched = False
        if can_switch and best_pair.name != self._active_pair.name:
            self._active_pair = best_pair
            self._active_since = 0
            switched = True
        else:
            self._active_since += 1

        decision = {
            "timestamp": timestamp,
            "selected_pair": self._active_pair.name,
            "selected_estimator": self._active_pair.estimator_name,
            "selected_controller": self._active_pair.controller_name,
            "switched": switched,
            "scores": scores,
            "score_components": score_components,
            "market_features": dict(market_features),
            "performance_metrics": dict(performance_metrics),
        }
        self.decisions.append(decision)
        return self._active_pair


class BaseRuleBasedRouter(Router):
    """Explicit paper-name alias for the base rule-based router."""
