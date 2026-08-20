# Router Component Ablation Snapshot - 2026-07-04

Current router implementation snapshot for the next ablation sequence.

## Baseline Run Contract

- OOS window: 2023-02-10 to 2026-02-10.
- Start from the full router pool, not the reduced 12-pair pool.
- Exclusions are hard removals: excluded pairs must be removed from the pair list before pair result loading, train metrics, recent rankings, AI prompts, selection, and diagnostics.
- Use precomputed AI regime series unless that component is the ablation target.
- Current default NVIDIA model path remains configurable.
- Current live-call JSON caps:
  - default max output tokens: 128
  - switch decision max output tokens: 64
  - selection max output tokens: 96
- Current runner supports `--api-key-env` and `--request-min-interval-seconds`.

## Current Router Components

1. Precomputed AI regime series.
2. AI switch-review layer.
3. AI pair-selection layer.
4. Active-regime suitability ranking.
5. Recent 100/60/20-day pair ranking context.
6. Deterministic train-window baseline champion.
7. RV22 naive-scaling benchmark comparison.
8. Active-pair exclusion from switch candidates.

## Proposed Four One-Component Ablations

Each run should start from the same full hard-filtered baseline pool. Ablations are not cumulative unless explicitly requested.

1. Remove active-regime suitability ranking.
2. Remove recent 100/60/20-day pair ranking context.
3. Remove deterministic train-window baseline champion.
4. Remove RV22 naive-scaling benchmark comparison.

