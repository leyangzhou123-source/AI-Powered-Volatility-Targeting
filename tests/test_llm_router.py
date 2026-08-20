"""Mock tests for the LLM router."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jpm-project-mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from src.router.llm_router import LLMRouter
from src.router.router import Router
from src.router.strategy_pair import StrategyPair
from scripts.router.calibrate_llm_router_events import build_summary


class DummyEstimator:
    pass


class DummyController:
    pass


class ConstantWeight:
    pass


class CVaRESTargeting:
    pass


class LassoVolatility:
    pass


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeUrlopenSequence:
    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)

    def __call__(self, *args, **kwargs):
        if not self.payloads:
            raise AssertionError("No fake response payloads left.")
        return FakeResponse(self.payloads.pop(0))


def build_router(**params) -> LLMRouter:
    pairs = [
        StrategyPair("safe_pair", DummyEstimator(), DummyController()),
        StrategyPair("risk_pair", DummyEstimator(), DummyController()),
    ]
    base_params = {"default_pair": "safe_pair", "sticky_period": 1, "fail_open": True}
    base_params.update(params)
    return LLMRouter(pairs, params=base_params)


class LLMRouterTests(unittest.TestCase):
    def test_base_router_excludes_estimators_and_controllers(self):
        pairs = [
            StrategyPair("safe_pair", DummyEstimator(), DummyController()),
            StrategyPair("constant_pair", DummyEstimator(), ConstantWeight()),
            StrategyPair("cvar_pair", DummyEstimator(), CVaRESTargeting()),
            StrategyPair("lasso_pair", LassoVolatility(), DummyController()),
        ]
        router = Router(
            pairs,
            params={
                "default_pair": "constant_pair",
                "excluded_estimators": ["lasso"],
                "excluded_controllers": ["constant", "cvar"],
            },
        )

        self.assertEqual([pair.name for pair in router.pairs], ["safe_pair"])
        self.assertEqual(router.default_pair.name, "safe_pair")

    def test_valid_mock_response_selects_llm_pair(self):
        router = build_router(api_key="test-key")
        payload = {
            "output_text": json.dumps(
                {
                    "pair": "risk_pair",
                    "reason": "Higher score with acceptable diagnostics.",
                    "confidence": 0.72,
                }
            )
        }

        with patch("src.router.llm_router.urlopen", return_value=FakeResponse(payload)):
            pair = router.select(
                market_features={"vol_regime": "mid", "rolling_vol": 0.14},
                diagnostics={},
                performance_metrics={"obs": 20, "rolling_sharpe": 0.8, "drawdown": 0.03},
                timestamp="mock",
            )

        self.assertEqual(pair.name, "risk_pair")
        self.assertTrue(router.decisions[-1]["llm_used"])
        self.assertEqual(router.decisions[-1]["llm_response"]["pair"], "risk_pair")
        self.assertIsNone(router.decisions[-1]["llm_error"])

    def test_empty_key_falls_back_without_crashing(self):
        router = build_router(api_key="")

        pair = router.select(
            market_features={"vol_regime": "low"},
            diagnostics={},
            performance_metrics={"obs": 0},
            timestamp="mock",
        )

        self.assertEqual(pair.name, "safe_pair")
        self.assertFalse(router.decisions[-1]["llm_used"])
        self.assertIn("API key is empty", router.decisions[-1]["llm_error"])

    def test_regime_suitability_affects_fallback_score(self):
        router = build_router(
            api_key="",
            use_heuristic_regime_bias=False,
            regime_bias={"high": {"risk_pair": 0.2}},
            estimator_regime_bias={"high": {"DummyEstimator": 0.3}},
            controller_regime_bias={"high": {"DummyController": 0.1}},
        )

        pair = router.select(
            market_features={"vol_regime": "high"},
            diagnostics={},
            performance_metrics={"obs": 0},
            timestamp="mock",
        )
        components = router.decisions[-1]["score_components"]["risk_pair"]

        self.assertEqual(pair.name, "risk_pair")
        self.assertAlmostEqual(components["pair_regime_bias"], 0.2)
        self.assertAlmostEqual(components["estimator_regime_bias"], 0.3)
        self.assertAlmostEqual(components["controller_regime_bias"], 0.1)
        self.assertAlmostEqual(components["regime_bias"], 0.6)

    def test_unknown_llm_pair_falls_back(self):
        router = build_router(api_key="test-key")
        payload = {"output_text": json.dumps({"pair": "not_a_real_pair", "reason": "bad"})}

        with patch("src.router.llm_router.urlopen", return_value=FakeResponse(payload)):
            pair = router.select(
                market_features={"vol_regime": "high"},
                diagnostics={},
                performance_metrics={"obs": 0},
                timestamp="mock",
            )

        self.assertEqual(pair.name, "safe_pair")
        self.assertFalse(router.decisions[-1]["llm_used"])
        self.assertIn("unknown pair", router.decisions[-1]["llm_error"])

    def test_nvidia_chat_completions_response_shape(self):
        router = build_router(
            api_key="test-key",
            provider="nvidia",
            api_format="chat_completions",
            model="openai/gpt-oss-120b",
        )
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"pair": "risk_pair", "reason": "chat completion", "confidence": 0.6}
                        )
                    }
                }
            ]
        }

        with patch("src.router.llm_router.urlopen", return_value=FakeResponse(payload)):
            pair = router.select(
                market_features={"vol_regime": "mid"},
                diagnostics={},
                performance_metrics={"obs": 5},
                timestamp="mock",
            )

        self.assertEqual(pair.name, "risk_pair")
        self.assertTrue(router.decisions[-1]["llm_used"])
        self.assertEqual(router.endpoint, "https://integrate.api.nvidia.com/v1/chat/completions")

    def test_diversity_guard_uses_close_alternative(self):
        router = build_router(
            api_key="test-key",
            use_heuristic_regime_bias=False,
            regime_bias={"high": {"safe_pair": 0.2, "risk_pair": 0.19}},
            max_consecutive_pair_calls=1,
            diversity_score_margin=0.05,
        )
        payload = {"output_text": json.dumps({"pair": "safe_pair", "reason": "repeat", "confidence": 0.8})}

        with patch("src.router.llm_router.urlopen", return_value=FakeResponse(payload)):
            first = router.select({"vol_regime": "high"}, {}, {"obs": 0}, timestamp="mock-1")
            second = router.select({"vol_regime": "high"}, {}, {"obs": 0}, timestamp="mock-2")

        self.assertEqual(first.name, "safe_pair")
        self.assertEqual(second.name, "risk_pair")
        self.assertIn("diversity_guard", router.decisions[-1]["llm_diversity_override"])

    def test_intraday_rv_event_mode_waits_for_curve_event(self):
        router = build_router(
            api_key="test-key",
            decision_mode="intraday_rv_event",
            always_call_first=False,
            min_decision_gap=1,
            rv_zscore_trigger=1.5,
        )
        payload = {"output_text": json.dumps({"pair": "risk_pair", "reason": "rv shock", "confidence": 0.8})}

        with patch("src.router.llm_router.urlopen", return_value=FakeResponse(payload)):
            first = router.select(
                {"vol_regime": "middle", "intraday_realized_vol": {"zscore": 0.2}},
                {},
                {"obs": 0},
                timestamp="mock-1",
            )
            second = router.select(
                {"vol_regime": "middle", "intraday_realized_vol": {"zscore": 2.0}},
                {},
                {"obs": 0},
                timestamp="mock-2",
            )

        self.assertEqual(first.name, "safe_pair")
        self.assertEqual(second.name, "risk_pair")
        self.assertFalse(router.decisions[0]["llm_used"])
        self.assertTrue(router.decisions[1]["llm_used"])
        self.assertIn("rv_zscore", router.decisions[1]["llm_call_reason"])

    def test_candidate_payload_includes_trailing_pair_performance(self):
        router = build_router(api_key="")
        _, scores, _ = router._fallback_pair({"vol_regime": "middle"}, {}, {"obs": 0})
        payload = json.loads(
            router._candidate_payload(
                {"vol_regime": "middle"},
                {},
                {
                    "obs": 0,
                    "risk_pair": {
                        "trailing_63d": {
                            "trailing_return": 0.04,
                            "rolling_sharpe": 1.2,
                            "drawdown": 0.03,
                        }
                    },
                    "benchmark": {"trailing_63d": {"trailing_return": 0.02}},
                },
                scores,
            )
        )

        risk = next(row for row in payload["candidates"] if row["name"] == "risk_pair")
        self.assertEqual(risk["trailing_performance"]["trailing_63d"]["trailing_return"], 0.04)
        self.assertEqual(payload["performance_metrics"]["benchmark"]["trailing_63d"]["trailing_return"], 0.02)
        self.assertNotIn("risk_pair", payload["performance_metrics"])

    def test_benchmark_underperformance_triggers_llm_review(self):
        router = build_router(
            api_key="test-key",
            decision_mode="intraday_rv_event",
            always_call_first=False,
            min_decision_gap=1,
            benchmark_underperformance_trigger=0.02,
        )
        router._active_pair = router.pairs[0]
        router._step = 1

        should_call, reason = router._should_call_llm(
            {"vol_regime": "middle", "intraday_realized_vol": {"zscore": 0.1}},
            {
                "obs": 80,
                "drawdown": 0.01,
                "safe_pair": {"trailing_63d": {"trailing_return": 0.01, "drawdown": 0.01}},
                "benchmark": {"trailing_63d": {"trailing_return": 0.04}},
            },
        )

        self.assertTrue(should_call)
        self.assertIn("benchmark_underperformance", reason)

    def test_two_stage_decision_payload_hides_candidate_information(self):
        router = build_router(two_stage_decision=True, review_interval=20)
        router._active_pair = router.pairs[0]
        router._selected_pair_history = ["safe_pair"] * 80 + ["risk_pair"] * 20
        payload = json.loads(
            router._decision_payload(
                {"vol_regime": "middle", "intraday_realized_vol": {"curve": [0.1, 0.12]}},
                {
                    "obs": 80,
                    "drawdown": 0.03,
                    "trailing_10d_return": -0.01,
                    "trailing_63d_return": 0.01,
                    "safe_pair": {
                        "trailing_10d": {
                            "trailing_return": -0.02,
                            "drawdown": 0.03,
                            "turnover": 0.01,
                        },
                        "trailing_63d": {"trailing_return": 0.01},
                    },
                    "benchmark": {
                        "trailing_10d": {"trailing_return": 0.01},
                        "trailing_63d": {"trailing_return": 0.04},
                    },
                },
            )
        )

        self.assertIn("intraday_realized_vol", payload)
        self.assertIn("strategy_performance", payload)
        self.assertIn("recent_10d_focus", payload)
        self.assertEqual(payload["recent_10d_focus"]["lookback"], "trailing_10d")
        self.assertEqual(payload["recent_10d_focus"]["return_minus_benchmark"], -0.03)
        self.assertEqual(payload["strategy_performance"]["trailing_10d_return"], -0.01)
        self.assertIn("benchmark", payload)
        self.assertIn("pair_concentration", payload)
        self.assertNotIn("candidates", payload)
        self.assertNotIn("safe_pair", json.dumps(payload["strategy_performance"]))

    def test_switch_selection_payload_excludes_active_pair(self):
        router = build_router(api_key="", candidate_top_n=10)
        router._active_pair = router.pairs[0]
        _, scores, _ = router._fallback_pair({"vol_regime": "middle"}, {}, {"obs": 0})
        payload = json.loads(
            router._candidate_payload(
                {"vol_regime": "middle"},
                {},
                {"obs": 0},
                scores,
                exclude_pair_names={"safe_pair"},
            )
        )

        names = [row["name"] for row in payload["candidates"]]
        self.assertNotIn("safe_pair", names)
        self.assertIn("risk_pair", names)
        self.assertEqual(payload["excluded_candidate_names"], ["safe_pair"])

    def test_two_stage_switch_rejects_active_pair_selection(self):
        router = build_router(
            api_key="test-key",
            two_stage_decision=True,
            review_interval=20,
            always_call_first=True,
            response_format={"type": "json_object"},
        )
        payloads = [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"action": "switch", "reason": "recent lag", "confidence": 0.8}
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"pair": "safe_pair", "reason": "same pair", "confidence": 0.7}
                            )
                        }
                    }
                ]
            },
        ]

        with patch("src.router.llm_router.urlopen", new=FakeUrlopenSequence(payloads)):
            pair = router.select(
                {"vol_regime": "middle", "intraday_realized_vol": {"curve": [0.1, 0.11]}},
                {},
                {"obs": 80, "drawdown": 0.02},
                timestamp="mock",
            )

        self.assertEqual(pair.name, "safe_pair")
        self.assertFalse(router.decisions[-1]["llm_used"])
        self.assertIn("excluded active pair", router.decisions[-1]["llm_error"])

    def test_selection_payload_includes_pair_concentration(self):
        router = build_router(api_key="", candidate_top_n=1, pair_concentration_threshold=0.7)
        router._selected_pair_history = ["safe_pair"] * 90 + ["risk_pair"] * 10
        _, scores, _ = router._fallback_pair({"vol_regime": "middle"}, {}, {"obs": 0})
        payload = json.loads(router._candidate_payload({"vol_regime": "middle"}, {}, {"obs": 0}, scores))

        self.assertTrue(payload["pair_concentration"]["is_concentrated"])
        self.assertEqual(payload["pair_concentration"]["dominant_pair"], "safe_pair")

    def test_risk_adjusted_candidate_ranking_prefers_target_vol_and_lower_drawdown(self):
        router = build_router(
            api_key="",
            candidate_top_n=1,
            candidate_rank_mode="risk_adjusted",
            llm_drawdown_penalty_prompt_weight=3.0,
            llm_vol_band_penalty_prompt_weight=10.0,
            llm_return_reward_prompt_weight=0.75,
            prompt_history_lookbacks=["trailing_10d"],
        )
        _, scores, _ = router._fallback_pair({"vol_regime": "middle"}, {}, {"obs": 0})
        payload = json.loads(
            router._candidate_payload(
                {"vol_regime": "middle"},
                {},
                {
                    "obs": 80,
                    "safe_pair": {
                        "trailing_10d": {
                            "trailing_return": 0.01,
                            "drawdown": 0.01,
                            "realized_vol": 0.10,
                            "turnover": 0.01,
                        },
                    },
                    "risk_pair": {
                        "trailing_10d": {
                            "trailing_return": 0.05,
                            "drawdown": 0.12,
                            "realized_vol": 0.16,
                            "turnover": 0.01,
                        },
                    },
                    "benchmark": {"trailing_10d": {"trailing_return": 0.0}},
                },
                scores,
            )
        )

        names = [row["name"] for row in payload["candidates"]]
        self.assertEqual(names, ["safe_pair"])
        self.assertTrue(payload["candidates"][0]["risk_target_check"]["passes_primary_vol_band"])

    def test_hard_risk_filter_removes_high_vol_high_drawdown_candidates(self):
        router = build_router(
            api_key="",
            candidate_top_n=10,
            candidate_hard_risk_filter=True,
            candidate_risk_filter_min_count=1,
            llm_hard_vol_max=0.12,
            llm_hard_drawdown_max=0.10,
            prompt_history_lookbacks=["trailing_63d"],
        )
        _, scores, _ = router._fallback_pair({"vol_regime": "middle"}, {}, {"obs": 0})
        payload = json.loads(
            router._candidate_payload(
                {"vol_regime": "middle"},
                {},
                {
                    "obs": 80,
                    "safe_pair": {
                        "trailing_63d": {
                            "trailing_return": 0.01,
                            "drawdown": 0.02,
                            "realized_vol": 0.10,
                            "turnover": 0.01,
                        },
                    },
                    "risk_pair": {
                        "trailing_63d": {
                            "trailing_return": 0.05,
                            "drawdown": 0.13,
                            "realized_vol": 0.16,
                            "turnover": 0.01,
                        },
                    },
                    "benchmark": {"trailing_63d": {"trailing_return": 0.0}},
                },
                scores,
            )
        )

        names = [row["name"] for row in payload["candidates"]]
        self.assertEqual(names, ["safe_pair"])
        self.assertTrue(payload["hard_risk_filter"]["applied"])
        self.assertEqual(payload["hard_risk_filter"]["filtered_count"], 1)

    def test_two_stage_switch_then_selection(self):
        router = build_router(
            api_key="test-key",
            two_stage_decision=True,
            review_interval=20,
            always_call_first=True,
            response_format={"type": "json_object"},
        )
        payloads = [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"action": "switch", "reason": "benchmark lag", "confidence": 0.8}
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"pair": "risk_pair", "reason": "beats benchmark net", "confidence": 0.7}
                            )
                        }
                    }
                ]
            },
        ]

        with patch("src.router.llm_router.urlopen", new=FakeUrlopenSequence(payloads)):
            pair = router.select(
                {"vol_regime": "middle", "intraday_realized_vol": {"curve": [0.1, 0.11]}},
                {},
                {
                    "obs": 80,
                    "drawdown": 0.02,
                    "risk_pair": {"trailing_63d": {"trailing_return": 0.05, "turnover": 0.01}},
                    "benchmark": {"trailing_63d": {"trailing_return": 0.03}},
                },
                timestamp="mock",
            )

        self.assertEqual(pair.name, "risk_pair")
        self.assertTrue(router.decisions[-1]["llm_used"])
        self.assertEqual(router.decisions[-1]["llm_response"]["decision"]["action"], "switch")
        self.assertEqual(router.decisions[-1]["llm_response"]["selection"]["pair"], "risk_pair")

    def test_ai_event_calibration_summary_does_not_grid_thresholds(self):
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {
                "realized_vol": [0.1 + (i % 7) * 0.01 for i in range(40)],
                "coverage": [1.0] * 40,
            },
            index=idx,
        )

        summary = build_summary(df, lookback=21)

        self.assertIn("abs_zscore_quantiles", summary)
        self.assertIn("recent_curve_tail", summary)
        self.assertNotIn("event_count_candidates_near_18_per_year", summary)
        self.assertNotIn("shock_days_z1p5_or_change25", summary)


if __name__ == "__main__":
    unittest.main()
