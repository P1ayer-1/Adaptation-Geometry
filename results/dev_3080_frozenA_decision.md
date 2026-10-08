# Decision memo — experiment `dev_3080_frozenA`

_Generated automatically by `uag analyze` (src/uag/analysis.py) on 2026-10-03 00:13:55 at commit `03b2af1feb38`. Do not edit by hand; rerun the analysis instead._

## Setup

- Bases: `dev_qwen2.5-0.5b` (qwen2, rev `local`), `dev_llama3.2-1b` (llama, rev `local`)
- Tasks: `T3_paraphrase`, `T7_python`, `T9_clinical`, `T10_format`, `T4_json`, `T8_arithmetic`
- Seeds: [0, 1]; LoRA r=16, alpha=32, modules=['q', 'k', 'v', 'o', 'up', 'down', 'gate']
- Evaluation split: `test` (first 200 examples)
- Held-out splits: `holdout_T4_T8` → ['T4_json', 'T8_arithmetic']

## Predeclared gate

Declared 2026-10-02T23:07:04 (sha256 `713f7bc334aa`), before held-out evaluation.

1. At least **1** heterogeneous ordered base pairs show positive, seed-repeatable held-out transfer on a majority of eligible tasks;
2. median RecoveredLift of `svd_procrustes` ≥ **0.3**;
3. `svd_procrustes` beats the strongest non-learned baseline (random, random_coords, target_mean) with a paired-bootstrap 95% CI excluding zero (resampling identical source×target×task cells).

Lift is measured against the **fewshot** base (5 worked training examples in the prompt, shot seed 0). Cells whose mean direct-LoRA lift is below 0.05 are ineligible (ratio unstable).

## Verdict

**NO-GO** — see decision tree (spec §19)

| Criterion | Value | Threshold | Passed |
|---|---|---|---|
| Heterogeneous pairs passing | 0 | ≥ 1 | False |
| Median RecoveredLift | -0.072 | ≥ 0.3 | False |
| Beats strongest baseline | Δ=-0.012 [-0.036, 0.000] vs `target_mean` (n=4) | CI > 0 | False |

### Ordered pairs (heterogeneous only)

| Source → Target | Eligible tasks | Positive | Passes |
|---|---|---|---|
| dev_llama3.2-1b → dev_qwen2.5-0.5b | 2 | 0 | False |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | 2 | 0 | False |

### Baselines (mean held-out score over cells shared with the primary method)

| Baseline | Cells | Mean score |
|---|---|---|
| random | 4 | 0.028 |
| random_coords | 4 | 0.026 |
| target_mean | 4 | 0.044 |

## Headroom: raw base vs direct LoRA (every base × task cell)

Lift = direct − fewshot base. Flags: **ceiling** = a base scores ≥ 0.9 (no room to measure transfer); **format** = the few-shot prompt alone recovers ≥ 50% of the zero-shot-referenced lift (the adapter mostly teaches output format).

| Base | Task | Metric | Zero-shot | Few-shot | Direct (mean ± sd over seeds) | Lift | Eligible | Flags |
|---|---|---|---|---|---|---|---|---|
| dev_qwen2.5-0.5b | T3_paraphrase | accuracy | 0.650 | 0.640 | 0.960 ± 0.014 (n=2) | 0.320 | True | – |
| dev_qwen2.5-0.5b | T7_python | unit_test_pass | 0.030 | 0.090 | 0.988 ± 0.011 (n=2) | 0.898 | True | – |
| dev_qwen2.5-0.5b | T9_clinical | exact_match | 0.000 | 0.050 | 1.000 ± 0.000 (n=2) | 0.950 | True | – |
| dev_qwen2.5-0.5b | T10_format | format_v2 | 0.000 | 0.000 | 0.952 ± 0.018 (n=2) | 0.952 | True | – |
| dev_qwen2.5-0.5b | T4_json | json_exact | 0.000 | 0.065 | 0.970 ± 0.028 (n=2) | 0.905 | True | – |
| dev_qwen2.5-0.5b | T8_arithmetic | exact_number | 0.000 | 0.060 | 0.897 ± 0.053 (n=2) | 0.837 | True | – |
| dev_llama3.2-1b | T3_paraphrase | accuracy | 0.445 | 0.585 | 0.965 ± 0.028 (n=2) | 0.380 | True | – |
| dev_llama3.2-1b | T7_python | unit_test_pass | 0.040 | 0.555 | 0.998 ± 0.004 (n=2) | 0.443 | True | format |
| dev_llama3.2-1b | T9_clinical | exact_match | 0.000 | 0.135 | 0.990 ± 0.014 (n=2) | 0.855 | True | – |
| dev_llama3.2-1b | T10_format | format_v2 | 0.000 | 0.005 | 0.980 ± 0.000 (n=2) | 0.975 | True | – |
| dev_llama3.2-1b | T4_json | json_exact | 0.000 | 0.385 | 0.998 ± 0.004 (n=2) | 0.613 | True | – |
| dev_llama3.2-1b | T8_arithmetic | exact_number | 0.110 | 0.170 | 0.905 ± 0.007 (n=2) | 0.735 | True | – |

## Held-out transfer cells (seed-averaged)

RecoveredLift = (S_transfer − S_base) / (S_direct − S_base) with S_base the fewshot base; '–' marks ineligible cells.

| Source → Target | Split | Task | Method | S_transfer | RecoveredLift (mean ± sd) | Δ rel. err | Seeds |
|---|---|---|---|---|---|---|---|
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | random | 0.000 | -0.072 ± 0.002 | 1.406 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | random_coords | 0.000 | -0.072 ± 0.002 | 1.406 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | svd_linear | 0.000 | -0.072 ± 0.002 | 1.026 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | svd_procrustes | 0.000 | -0.072 ± 0.002 | 1.019 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | target_mean | 0.000 | -0.072 ± 0.002 | 1.058 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | random | 0.000 | -0.072 ± 0.005 | 1.340 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | random_coords | 0.000 | -0.072 ± 0.005 | 1.340 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | svd_linear | 0.000 | -0.072 ± 0.005 | 1.025 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | svd_procrustes | 0.000 | -0.072 ± 0.005 | 1.019 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | target_mean | 0.000 | -0.072 ± 0.005 | 1.059 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | random | 0.000 | -0.629 ± 0.004 | 1.436 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | random_coords | 0.000 | -0.629 ± 0.004 | 1.436 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | svd_linear | 0.000 | -0.629 ± 0.004 | 1.051 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | svd_procrustes | 0.000 | -0.629 ± 0.004 | 1.034 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | target_mean | 0.000 | -0.629 ± 0.004 | 1.066 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | random | 0.110 | -0.082 ± 0.039 | 1.533 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | random_coords | 0.103 | -0.092 ± 0.004 | 1.533 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | svd_linear | 0.100 | -0.095 ± 0.001 | 1.067 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | svd_procrustes | 0.130 | -0.054 ± 0.019 | 1.050 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | target_mean | 0.177 | 0.010 ± 0.034 | 1.086 | 2 |

## Exclusion / failure log

No exclusions recorded.

## Reproduce

```bash
uag analyze --experiment <config for dev_3080_frozenA>
```
