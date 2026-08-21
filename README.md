# JPM Volatility Targeting

Research code for volatility-targeted portfolios, estimator-controller routing,
and AI-assisted regime selection. The project supports both single-asset SP500
experiments and multi-asset portfolio experiments.

## What This Project Does

This repository studies volatility control as a routing problem:

- Estimate risk using volatility or covariance models.
- Convert risk estimates into exposure or portfolio weights with controllers.
- Backtest estimator-controller pairs with transaction costs and a risk-free leg.
- Route among candidate pairs with rule-based, mixture-of-experts, contextual
  bandit, LLM, and AI-regime routers.
- Evaluate performance, drawdown, turnover, regime behavior, and pair selection.

The current codebase contains two main paths:

- **SP500 / single asset:** one risky asset plus cash, using `VolTargetEngine`.
- **Multi asset / portfolio:** wide asset-return matrix plus cash, using
  `MultiAssetVolTargetEngine`.

## Repository Layout

```text
.
├── configs/
│   └── strategies/              # YAML strategy and router configs
├── scripts/                     # CLI entry points for runs, tuning, reports
│   └── router/                  # Router-specific workflows and experiments
├── src/
│   ├── backtest/                # Single-asset engine and shared engine exports
│   ├── controllers/             # Single-asset exposure controllers
│   ├── estimators/              # Single-asset volatility estimators
│   ├── evaluation/              # Metrics and router logging helpers
│   ├── multi_asset/             # Portfolio covariance estimators/controllers/engine
│   └── router/                  # Rule, bandit, MoE, LLM, and AI-regime routers
├── tests/                       # Unit tests
├── data/                        # Local raw/processed data, ignored by Git
└── results/                     # Local backtest outputs, ignored by Git
```

## Installation

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional dependency groups are defined in `pyproject.toml`:

```bash
python -m pip install -e ".[ar,garch,ml,hmm]"
```

Use `.[all]` if you want every optional research dependency, including data
vendor tooling.

## Data

Large data files are intentionally not tracked by Git. Put local datasets under
`data/processed/` or `data/raw/`.

Common files referenced by configs and scripts:

- `data/processed/Master_Dataset.parquet`
- `data/processed/SP500_Intraday_RealizedVol.parquet`
- `data/processed/emm_daily_log_returns_yahoo_20220210_20260210.parquet`
- `data/processed/VIX_Daily_Processed.parquet`
- `data/processed/credit_spreads_2021_2026.parquet`

Generated outputs are written under `results/` and are also ignored by Git.

## Quick Start

Run one strategy config:

```bash
python scripts/run_backtests.py --strategy configs/strategies/realized_vol.yaml
```

Run all strategy YAMLs:

```bash
python scripts/run_backtests.py
```

Run the test suite:

```bash
pytest
```

If `pytest` is unavailable, the current standard-library fallback for the
included router tests is:

```bash
python -m unittest tests.test_llm_router
```

## SP500 / Single-Asset Stack

Single-asset strategies are configured with YAML files in
`configs/strategies/`. Most SP500 configs use:

- `data.symbol: SPY`
- `target_vol: 0.1`
- `rebalance_freq: daily`
- `roll_window: 252`
- `weight_min: 0.0`
- `weight_max: 1.5`
- `cost_bps: 5.0`

The engine is:

- `src.backtest.engine.VolTargetEngine`

Important SP500 configs:

- `configs/strategies/realized_vol.yaml`
- `configs/strategies/EWMA.yaml`
- `configs/strategies/AR1.yaml`
- `configs/strategies/AR2.yaml`
- `configs/strategies/lightgbx.yaml`
- `configs/strategies/rnn.yaml`
- `configs/strategies/router_master.yaml`

The master SP500 router config, `router_master.yaml`, contains the broad
estimator-controller universe for SPY routing.

Single-asset estimator code lives in `src/estimators/`, including:

- `RealizedVol`
- `EWMA`
- `AR1`
- `AR2`
- `GARCH`
- `GJRGARCH`
- `HARRV`
- `HARRVRates`
- `RSHARRates`
- `RegimeGJRGARCH`
- `HybridEWMARegime`
- `DynamicPrecisionEnsemble`
- `LightGBXVolatility`
- `RandomForestVolEstimator`
- `RNNVolatility`
- `XGB_VIX`
- `LassoVolatility`
- `NaiveVolEstimator`
- `BuyAndHold`

Single-asset controller code lives in `src/controllers/`, including:

- `NaiveScaling`
- `ConstantWeight`
- `VarianceScaling`
- `RegimeSwitchController`
- `VolTargetClip`
- `DrawdownBrake`
- `DrawdownModulatedController`
- `TrendFilter`
- `CVaRESTargeting`
- `HysteresisController`
- `PriorityStackController`

Example pair config:

```yaml
router:
  enabled: true
  default_pair: realized_vol__naive_scaling
  pairs:
    - name: realized_vol__naive_scaling
      estimator:
        class: src.estimators.RealizedVol
        params:
          lookback: 20
          vol_ann: 252
      controller:
        class: src.controllers.naive_scaling.NaiveScaling
        params: {}
```

## Multi-Asset Stack

Multi-asset strategies set:

```yaml
engine_mode: multi_asset
```

The engine is:

- `src.multi_asset.engine.MultiAssetVolTargetEngine`

The template config is:

- `configs/strategies/multi_asset_ewma_inverse_vol.yaml`

It uses:

```yaml
estimator:
  class: src.multi_asset.covariance_estimators.EWMACovariance
controller:
  class: src.multi_asset.controllers.InverseVolController
```

Multi-asset covariance estimators live in
`src/multi_asset/covariance_estimators.py`. The registry currently includes:

- `sample_covariance`
- `expanding_covariance`
- `ewma_covariance`
- `diagonal_ewma_covariance`
- `rolling_corr_ewma_vol`
- `shrunk_sample_covariance`
- `ledoit_wolf_covariance`
- `downside_covariance`
- `robust_median_covariance`
- `regime_switching_covariance`
- `vix_scaled_covariance`
- `pca_covariance`
- `dynamic_blend_covariance`

Multi-asset portfolio controllers live in `src/multi_asset/controllers.py`.
The registry currently includes:

- `equal_weight`
- `buy_and_hold`
- `inverse_vol`
- `minimum_variance`
- `vol_capped_min_variance`
- `equal_risk_contribution`
- `diversified_risk_parity`
- `momentum_tilt`
- `mean_variance`
- `regime_aware_risk_budget`
- `drawdown_brake_portfolio`
- `hysteresis_portfolio`

Generate and run the full multi-asset pair universe:

```bash
python scripts/run_multi_asset_pairs.py \
  --out-dir results/multi_asset_pairs \
  --data-path data/processed/emm_daily_log_returns_yahoo_20220210_20260210.parquet
```

This builds every registered covariance-estimator/portfolio-controller pair and
writes a `manifest.csv` that downstream routers can consume.

## Routers

Router code lives in `src/router/`.

Available router classes include:

- `Router`
- `BaseRuleBasedRouter`
- `RuleConstraintRouter`
- `ContextualBanditRouter`
- `MixtureOfExpertsRouter`
- `LLMRouter`
- `AIRegimeRouter`

### SP500 AI Regime Router

Code:

- `src/router/ai_regime_router.py`

Main workflow:

```bash
python scripts/router/run_ai_regime_router_all_pairs.py \
  --strategy configs/strategies/router_master.yaml \
  --output-dir results/evaluation/ai_regime_router_precomputed
```

This script loads `router_master.yaml`, injects:

```yaml
router:
  enabled: true
  type: ai_regime
  class: src.router.ai_regime_router.AIRegimeRouter
```

and configures AI/model settings, intraday realized-volatility features,
pair-history features, exclusions, and candidate selection limits.

Common defaults:

- provider: `nvidia`
- api format: `chat_completions`
- model: `openai/gpt-oss-120b`
- AI regime interval: `10`
- AI selection interval: `10`
- candidate top N: `6`
- regime suitability path: `results/evaluation/regime_suitability_all_pairs`
- precomputed regime path: `results/evaluation/ai_regime_series/ai_regime_10d.csv`

### Multi-Asset AI Portfolio Regime Router

Code:

- `src/router/ai_portfolio_regime_router.py`

Main workflow:

```bash
python scripts/router/run_ai_portfolio_regime_router.py \
  --manifest results/multi_asset_tuned_pairs_vol10/manifest.csv \
  --out results/multi_asset_tuned_pairs_vol10/ai_portfolio_router_live.parquet \
  --window 63 \
  --regime-rank-window 63 \
  --params-json '{}'
```

There is no dedicated static YAML for this router. It is configured through
CLI arguments and `--params-json`. Important parameters include:

- `provider`
- `api_format`
- `model`
- `ai_enabled`
- `precomputed_regime_path`
- `initial_pair`
- `candidate_top_n`
- `regime_rank_top_n`
- `recent_rank_windows`
- `overall_rank_window`
- `switch_review_interval`
- `selection_interval`
- `included_pairs`
- `excluded_pairs_containing`

The model-sweep workflow in `scripts/router/run_multi_asset_ai_model_sweep.py`
shows a production-style parameter bundle for experiments.

## LLM And API Keys

API keys are not committed. Use environment variables:

```bash
export OPENAI_API_KEY="..."
export NVIDIA_API_KEY="..."
export NVAPI_KEY="..."
```

The simple assistant config is:

- `configs/ai_assistant.yaml`

Run it with:

```bash
python scripts/ai_assistant.py "generate a conservative router YAML spec"
```

Routers are designed to fail open when configured that way: if an API key is
missing or a response is invalid, they can fall back to deterministic scoring.

## Common Research Commands

Build regime suitability priors from rolling pair backtests:

```bash
python scripts/router/build_regime_suitability.py \
  --manifest results/all_estimator_controller_pairs/manifest.csv \
  --output-dir results/evaluation/regime_suitability_all_pairs \
  --top-n-bias 40
```

Calibrate LLM router event thresholds from SP500 intraday realized volatility:

```bash
python scripts/router/calibrate_llm_router_events.py \
  --rv-path data/processed/SP500_Intraday_RealizedVol.parquet \
  --output-dir results/evaluation/llm_router_event_calibration \
  --provider nvidia \
  --model openai/gpt-oss-120b
```

Tune non-AI router parameters:

```bash
python scripts/router/tune_router_params.py \
  --strategy configs/strategies/router_master.yaml \
  --pair-results-dir results/all_estimator_controller_pairs \
  --output-dir results/evaluation/router_param_tuning
```

Run a quick parameter grid:

```bash
python scripts/router/tune_router_params.py --quick \
  --param sticky_period=5,10,21 \
  --param lambda_risk=1.0,2.0,3.0
```

## Git Hygiene

The `.gitignore` excludes local credentials, generated data, generated results,
caches, binary research outputs, and temporary files. In particular:

- `.env` and `.env.*` are ignored.
- `data/*` is ignored except `.gitkeep`.
- `results/*` is ignored except `.gitkeep`.
- parquet, pickle, HDF5, spreadsheet, and joblib artifacts are ignored.

This keeps the repository suitable for GitHub while allowing large experiments
to run locally.

## License

This project is licensed under the terms in `LICENSE`.
