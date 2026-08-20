# JPM-Volatility-Targeting

.
├── configs/                # Strategy configuration files
│   └── strategies/         # Model-specific YAML configs (GARCH, EWMA, AR1, etc.)
├── src/                    # Core source code
│   ├── estimators/         # Volatility forecasting models (Realized Vol, EWMA, GJRGARCH, GARCH, etc.)
│   ├── controllers/        # Exposure control & leverage logic (Naive Scaling, Constant Weight)
│   ├── backtest/           # Backtesting engine core (engine.py, base classes)
│   ├── data/               # Data ingestion & ETL pipeline (importers, processors)
│   └── evaluation/         # Performance metrics & diagnostic tools
├── scripts/                # Execution & CLI scripts
│   ├── run_backtests.py    # Batch execution of backtesting suites
│   ├── analysus_result.py  # Automated 4-dimensional diagnostic report generation
│   ├── run_processors.py   # Execution of data cleaning & processing pipelines
│   └── tune_strategy.py    # Hyperparameter optimization & strategy tuning
├── data/                   # Data storage for Raw and Processed (Parquet) datasets
└── Literature-Review/      # Academic papers on Volatility-Managed Portfolios (Moreira, Harvey, etc.)


## Setup

Create an environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Data files, generated backtest outputs, caches, and local credentials are intentionally
ignored by Git. Keep raw/processed datasets under `data/` and generated reports under
`results/` locally.


## Simple AI Assistant

The project includes a minimal OpenAI-backed assistant configured in
`configs/ai_assistant.yaml`. The API key is intentionally blank:

```yaml
api_key: ""
model: gpt-5-mini
```

Run it with:

```bash
python scripts/ai_assistant.py "generate a conservative router YAML spec"
```

When you are ready to make real API calls, put your key in the config or export
`OPENAI_API_KEY`.

To let the assistant choose estimator-controller pairs inside the backtest loop,
set a strategy config's router type to `llm`:

```yaml
router:
  enabled: true
  type: llm
  default_pair: ewma__constant_weight
  params:
    api_key: ""
    model: gpt-5-mini
    sticky_period: 5
    decision_interval: 5
    max_calls: 0  # 0 means unlimited
    candidate_top_n: 40
    max_consecutive_pair_calls: 2
    diversity_score_margin: 0.12
    fail_open: true
```

The LLM may only choose from the configured `router.pairs`. If the token is
blank or the response is invalid, the router falls back to the transparent base
score so the backtest can still run.

Build regime suitability scores from rolling pair backtests:

```bash
python scripts/router/build_regime_suitability.py \
  --manifest results/all_estimator_controller_pairs/manifest.csv \
  --output-dir results/evaluation/regime_suitability_all_pairs \
  --top-n-bias 40
```

This writes pair, estimator, and controller suitability tables plus a
`router_regime_bias.yaml` snippet. Treat these as research priors unless they
are produced inside a nested train/test protocol.

Use those scores in any router by adding:

```yaml
router:
  params:
    regime_suitability_path: results/evaluation/regime_suitability_all_pairs
    use_heuristic_regime_bias: false
```

The shared router score will combine any available pair, estimator, and
controller suitability for the current regime. The LLM router also includes
these suitability values in the candidate payload sent to the model.

Use intraday-realized-volatility event timing for the LLM router:

```yaml
intraday_realized_vol:
  enabled: true
  path: data/processed/SP500_Intraday_RealizedVol.parquet
  lookback: 21

router:
  type: llm
  params:
    decision_mode: intraday_rv_event
    always_call_first: true
    min_decision_gap: 21
    rv_zscore_trigger: 1.5
    rv_change_trigger: 0.25
    rv_percentile_trigger: 0.9
```

In this mode the router sends the recent intraday-derived realized-volatility
curve and summary statistics to the LLM only when the curve shows a meaningful
shock, percentile extreme, or regime change after the minimum gap.

Ask an AI to calibrate those event thresholds from historical intraday-derived
realized volatility:

```bash
python scripts/router/calibrate_llm_router_events.py \
  --rv-path data/processed/SP500_Intraday_RealizedVol.parquet \
  --output-dir results/evaluation/llm_router_event_calibration \
  --provider nvidia \
  --model openai/gpt-oss-120b
```

Use `--no-api` to write only the summary and prompt. This calibration is a
single AI policy recommendation from historical intraday-RV distribution data;
the script does not search threshold grids or optimize policies inside the
backtest loop.

Tune the remaining non-AI router parameters deterministically:

```bash
python scripts/router/tune_router_params.py \
  --strategy configs/strategies/router_master.yaml \
  --pair-results-dir results/all_estimator_controller_pairs \
  --output-dir results/evaluation/router_param_tuning
```

This uses precomputed pair backtests and a walk-forward objective. It rejects
API, model, and intraday-RV event keys so Stage 1 remains AI-calibrated. Use
`--quick` for a small smoke grid, or add explicit grids such as:

```bash
python scripts/router/tune_router_params.py --quick \
  --param sticky_period=5,10,21 \
  --param lambda_risk=1.0,2.0,3.0
```
