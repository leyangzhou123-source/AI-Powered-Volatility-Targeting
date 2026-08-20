"""Registries for ensemble estimators and regime recognizers."""

from __future__ import annotations

import importlib
from typing import Any

from src.ensemble.regime.bayesian_change_point_recognizer import BayesianChangePointRecognizer
from src.ensemble.regime.change_point_recognizer import ChangePointRecognizer
from src.ensemble.regime.hmm_recognizer import HMMRecognizer
from src.ensemble.regime.markov_switching_garch_recognizer import MarkovSwitchingGARCHRecognizer
from src.ensemble.regime.markov_switching_recognizer import MarkovSwitchingRecognizer
from src.ensemble.regime.signal_rule_recognizer import SignalRuleRecognizer

ESTIMATOR_REGISTRY = {
    "ar1": "src.estimators.ar1.AR1",
    "ar2": "src.estimators.ar2.AR2",
    "ewma": "src.estimators.ewma.EWMA",
    "garch": "src.estimators.garch.GARCH",
    "garch_11": "src.estimators.garch.GARCH",
    "gjr_garch": "src.estimators.gjr_garch.GJRGARCH",
    "har_rv": "src.estimators.har_rv.HARRV",
    "lasso_vol": "src.estimators.lasso_vol.LassoVolatility",
    "lightgbx": "src.estimators.lightgbx_vol.LightGBXVolatility",
    "realized_vol": "src.estimators.realized_vol.RealizedVol",
    "rnn_vol": "src.estimators.rnn_vol.RNNVolatility",
    "wavelet": "src.estimators.wavelet.WaveletVolatility",
    "buy_and_hold": "src.estimators.buy_and_hold.BuyAndHold",
}

RECOGNIZER_REGISTRY = {
    "hmm": HMMRecognizer,
    "markov_switching": MarkovSwitchingRecognizer,
    "markov_switching_garch": MarkovSwitchingGARCHRecognizer,
    "change_point": ChangePointRecognizer,
    "bayesian_change_point": BayesianChangePointRecognizer,
    "signal_rule": SignalRuleRecognizer,
}



def _load_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_estimator(name: str, params: dict[str, Any] | None = None):
    key = str(name).lower()
    if key not in ESTIMATOR_REGISTRY:
        raise KeyError(f"Unknown estimator '{name}'. Known={sorted(ESTIMATOR_REGISTRY)}")
    cls = _load_class(ESTIMATOR_REGISTRY[key])

    est_params = dict(params or {})
    if key == "garch_11":
        est_params.setdefault("p", 1)
        est_params.setdefault("q", 1)

    return cls(est_params)


def create_recognizer(cfg: dict[str, Any] | None):
    cfg = dict(cfg or {})
    kind = str(cfg.get("type", "signal_rule")).lower()
    params = dict(cfg.get("params", {}))

    if kind == "signal_rule":
        params.setdefault("signals", cfg.get("signals", {}))
        params.setdefault("rules", cfg.get("rules", []))

    if kind not in RECOGNIZER_REGISTRY:
        raise KeyError(f"Unknown recognizer '{kind}'. Known={sorted(RECOGNIZER_REGISTRY)}")

    return RECOGNIZER_REGISTRY[kind](params=params)
