# Decision memo — experiment `dev_3080`

_Generated automatically by `uag analyze` (src/uag/analysis.py) on 2026-09-29 16:05:01 at commit `050c17a0c8e5`. Do not edit by hand; rerun the analysis instead._

## Setup

- Bases: `dev_qwen2.5-0.5b` (qwen2, rev `local`), `dev_llama3.2-1b` (llama, rev `local`)
- Tasks: `T1_sentiment`, `T4_json`, `T8_arithmetic`
- Seeds: [0]; LoRA r=16, alpha=32, modules=['q', 'k', 'v', 'o', 'up', 'down', 'gate']
- Evaluation split: `test` (first 100 examples)
- Held-out splits: `dev_holdout_T8` → ['T8_arithmetic']

## Predeclared gate

Declared 2026-09-29T16:03:01 (sha256 `f90c8e6d46f0`), before held-out evaluation.

1. At least **1** heterogeneous ordered base pairs show positive, seed-repeatable held-out transfer on a majority of eligible tasks;
2. median RecoveredLift of `svd_procrustes` ≥ **0.3**;
3. `svd_procrustes` beats the strongest non-learned baseline (identity, random) with a paired-bootstrap 95% CI excluding zero (resampling identical source×target×task cells).

Cells whose mean direct-LoRA lift is below 0.05 are ineligible (ratio unstable).

## Verdict

**GO** — proceed to Stage 1

| Criterion | Value | Threshold | Passed |
|---|---|---|---|
| Heterogeneous pairs passing | 2 | ≥ 1 | True |
| Median RecoveredLift | 0.463 | ≥ 0.3 | True |
| Beats strongest baseline | Δ=0.440 [0.170, 0.710] vs `random` (n=2) | CI > 0 | True |

### Ordered pairs (heterogeneous only)

| Source → Target | Eligible tasks | Positive | Passes |
|---|---|---|---|
| dev_llama3.2-1b → dev_qwen2.5-0.5b | 1 | 1 | True |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | 1 | 1 | True |

### Baselines (mean held-out score over cells shared with the primary method)

| Baseline | Cells | Mean score |
|---|---|---|
| identity | 0 | – |
| random | 2 | 0.125 |

## Headroom: raw base vs direct LoRA (every base × task cell)

| Base | Task | Metric | Base | Direct (mean ± sd over seeds) | Lift | Eligible |
|---|---|---|---|---|---|---|
| dev_qwen2.5-0.5b | T1_sentiment | accuracy | 0.950 | 1.000 ± 0.000 (n=1) | 0.050 | True |
| dev_qwen2.5-0.5b | T4_json | json_field_f1 | 0.015 | 1.000 ± 0.000 (n=1) | 0.985 | True |
| dev_qwen2.5-0.5b | T8_arithmetic | exact_number | 0.200 | 0.990 ± 0.000 (n=1) | 0.790 | True |
| dev_llama3.2-1b | T1_sentiment | accuracy | 0.900 | 1.000 ± 0.000 (n=1) | 0.100 | True |
| dev_llama3.2-1b | T4_json | json_field_f1 | 0.000 | 1.000 ± 0.000 (n=1) | 1.000 | True |
| dev_llama3.2-1b | T8_arithmetic | exact_number | 0.140 | 1.000 ± 0.000 (n=1) | 0.860 | True |

## Held-out transfer cells (seed-averaged)

RecoveredLift = (S_transfer − S_base) / (S_direct − S_base); '–' marks ineligible cells.

| Source → Target | Split | Task | Method | S_transfer | RecoveredLift (mean ± sd) | Δ rel. err | Seeds |
|---|---|---|---|---|---|---|---|
| dev_llama3.2-1b → dev_qwen2.5-0.5b | dev_holdout_T8 | T8_arithmetic | random | 0.100 | -0.127 ± 0.000 | 1.447 | 1 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | dev_holdout_T8 | T8_arithmetic | svd_procrustes | 0.270 | 0.089 ± 0.000 | 1.025 | 1 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | dev_holdout_T8 | T8_arithmetic | random | 0.150 | 0.012 ± 0.000 | 1.394 | 1 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | dev_holdout_T8 | T8_arithmetic | svd_procrustes | 0.860 | 0.837 ± 0.000 | 1.053 | 1 |

## Exclusion / failure log

| Kind | Reason | Detail |
|---|---|---|
| map | method_not_applicable | dev_qwen2.5-0.5b__to__dev_llama3.2-1b__dev_holdout_T8__identity |
| map | method_not_applicable | dev_llama3.2-1b__to__dev_qwen2.5-0.5b__dev_holdout_T8__identity |

## Reproduce

```bash
uag analyze --experiment <config for dev_3080>
```
