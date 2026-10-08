# Decision memo — experiment `dev_a100_v2_cmp`

_Generated automatically by `uag analyze` (src/uag/analysis.py) on 2026-10-03 01:00:32 at commit `34ee29e040fc`. Do not edit by hand; rerun the analysis instead._

## Setup

- Bases: `dev_qwen2.5-0.5b` (qwen2, rev `local`), `dev_llama3.2-1b` (llama, rev `local`)
- Tasks: `T3_paraphrase`, `T7_python`, `T9_clinical`, `T10_format`, `T4_json`, `T8_arithmetic`
- Seeds: [0, 1]; LoRA r=16, alpha=32, modules=['q', 'k', 'v', 'o', 'up', 'down', 'gate']
- Evaluation split: `test` (first 200 examples)
- Held-out splits: `holdout_T4_T8` → ['T4_json', 'T8_arithmetic']

## Predeclared gate

Declared 2026-10-03T00:15:07 (sha256 `713f7bc334aa`), before held-out evaluation.

1. At least **1** heterogeneous ordered base pairs show positive, seed-repeatable held-out transfer on a majority of eligible tasks;
2. median RecoveredLift of `svd_procrustes` ≥ **0.3**;
3. `svd_procrustes` beats the strongest non-learned baseline (random, random_coords, target_mean) with a paired-bootstrap 95% CI excluding zero (resampling identical source×target×task cells).

Lift is measured against the **fewshot** base (5 worked training examples in the prompt, shot seed 0). Cells whose mean direct-LoRA lift is below 0.05 are ineligible (ratio unstable).

## Verdict

**NO-GO** — see decision tree (spec §19)

| Criterion | Value | Threshold | Passed |
|---|---|---|---|
| Heterogeneous pairs passing | 0 | ≥ 1 | False |
| Median RecoveredLift | -0.074 | ≥ 0.3 | False |
| Beats strongest baseline | Δ=0.000 [0.000, 0.000] vs `random` (n=4) | CI > 0 | False |

### Ordered pairs (heterogeneous only)

| Source → Target | Eligible tasks | Positive | Passes |
|---|---|---|---|
| dev_llama3.2-1b → dev_qwen2.5-0.5b | 2 | 0 | False |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | 2 | 0 | False |

### Baselines (mean held-out score over cells shared with the primary method)

| Baseline | Cells | Mean score |
|---|---|---|
| random | 4 | 0.028 |
| random_coords | 4 | 0.019 |
| target_mean | 4 | 0.001 |

## Headroom: raw base vs direct LoRA (every base × task cell)

Lift = direct − fewshot base. Flags: **ceiling** = a base scores ≥ 0.9 (no room to measure transfer); **format** = the few-shot prompt alone recovers ≥ 50% of the zero-shot-referenced lift (the adapter mostly teaches output format).

| Base | Task | Metric | Zero-shot | Few-shot | Direct (mean ± sd over seeds) | Lift | Eligible | Flags |
|---|---|---|---|---|---|---|---|---|
| dev_qwen2.5-0.5b | T3_paraphrase | accuracy | 0.650 | 0.640 | 0.990 ± 0.000 (n=2) | 0.350 | True | – |
| dev_qwen2.5-0.5b | T7_python | unit_test_pass | 0.030 | 0.090 | 0.995 ± 0.000 (n=2) | 0.905 | True | – |
| dev_qwen2.5-0.5b | T9_clinical | exact_match | 0.000 | 0.050 | 1.000 ± 0.000 (n=2) | 0.950 | True | – |
| dev_qwen2.5-0.5b | T10_format | format_v2 | 0.000 | 0.000 | 0.985 ± 0.014 (n=2) | 0.985 | True | – |
| dev_qwen2.5-0.5b | T4_json | json_exact | 0.000 | 0.065 | 1.000 ± 0.000 (n=2) | 0.935 | True | – |
| dev_qwen2.5-0.5b | T8_arithmetic | exact_number | 0.000 | 0.060 | 0.903 ± 0.004 (n=2) | 0.843 | True | – |
| dev_llama3.2-1b | T3_paraphrase | accuracy | 0.445 | 0.585 | 0.962 ± 0.032 (n=2) | 0.377 | True | – |
| dev_llama3.2-1b | T7_python | unit_test_pass | 0.040 | 0.555 | 1.000 ± 0.000 (n=2) | 0.445 | True | format |
| dev_llama3.2-1b | T9_clinical | exact_match | 0.000 | 0.135 | 1.000 ± 0.000 (n=2) | 0.865 | True | – |
| dev_llama3.2-1b | T10_format | format_v2 | 0.000 | 0.005 | 0.985 ± 0.014 (n=2) | 0.980 | True | – |
| dev_llama3.2-1b | T4_json | json_exact | 0.000 | 0.385 | 1.000 ± 0.000 (n=2) | 0.615 | True | – |
| dev_llama3.2-1b | T8_arithmetic | exact_number | 0.110 | 0.170 | 0.920 ± 0.007 (n=2) | 0.750 | True | – |

## Held-out transfer cells (seed-averaged)

RecoveredLift = (S_transfer − S_base) / (S_direct − S_base) with S_base the fewshot base; '–' marks ineligible cells.

| Source → Target | Split | Task | Method | S_transfer | RecoveredLift (mean ± sd) | Δ rel. err | Seeds |
|---|---|---|---|---|---|---|---|
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | random | 0.000 | -0.070 ± 0.000 | 1.455 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | random_coords | 0.000 | -0.070 ± 0.000 | 1.455 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | svd_linear | 0.000 | -0.070 ± 0.000 | 1.000 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | svd_procrustes | 0.000 | -0.070 ± 0.000 | 1.000 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T4_json | target_mean | 0.000 | -0.070 ± 0.000 | 1.076 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | random | 0.000 | -0.071 ± 0.000 | 1.511 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | random_coords | 0.000 | -0.071 ± 0.000 | 1.511 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | svd_linear | 0.000 | -0.071 ± 0.000 | 1.000 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | svd_procrustes | 0.000 | -0.071 ± 0.000 | 1.000 | 2 |
| dev_llama3.2-1b → dev_qwen2.5-0.5b | holdout_T4_T8 | T8_arithmetic | target_mean | 0.000 | -0.071 ± 0.000 | 1.089 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | random | 0.000 | -0.626 ± 0.000 | 1.391 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | random_coords | 0.000 | -0.626 ± 0.000 | 1.391 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | svd_linear | 0.000 | -0.626 ± 0.000 | 1.001 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | svd_procrustes | 0.000 | -0.626 ± 0.000 | 1.001 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T4_json | target_mean | 0.000 | -0.626 ± 0.000 | 1.068 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | random | 0.113 | -0.076 ± 0.042 | 1.353 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | random_coords | 0.075 | -0.126 ± 0.046 | 1.353 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | svd_linear | 0.107 | -0.083 ± 0.024 | 1.001 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | svd_procrustes | 0.113 | -0.077 ± 0.005 | 1.001 | 2 |
| dev_qwen2.5-0.5b → dev_llama3.2-1b | holdout_T4_T8 | T8_arithmetic | target_mean | 0.005 | -0.220 ± 0.002 | 1.070 | 2 |

## Exclusion / failure log

No exclusions recorded.

## Reproduce

```bash
uag analyze --experiment <config for dev_a100_v2_cmp>
```
