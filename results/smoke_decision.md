# Decision memo — experiment `smoke`

_Generated automatically by `uag analyze` (src/uag/analysis.py) on 2026-09-29 20:06:50 at commit `4cdb53b828ed`. Do not edit by hand; rerun the analysis instead._

> **Pipeline validation only.** This experiment uses randomly initialised tiny bases; its numbers carry no scientific weight.

## Setup

- Bases: `tiny_llama_a1` (tiny_llama, rev `local`), `tiny_qwen2_b` (tiny_qwen2, rev `local`)
- Tasks: `T1_sentiment`, `T3_paraphrase`
- Seeds: [0, 1]; LoRA r=4, alpha=8, modules=['q', 'k', 'v', 'o', 'up', 'down', 'gate']
- Evaluation split: `test` (first 200 examples)
- Held-out splits: `toy_fitT1_predT3` → ['T3_paraphrase']; `toy_fitT3_predT1` → ['T1_sentiment']

## Predeclared gate

Declared 2026-09-29T20:06:43 (sha256 `6e8866e1724b`), before held-out evaluation.

1. At least **2** heterogeneous ordered base pairs show positive, seed-repeatable held-out transfer on a majority of eligible tasks;
2. median RecoveredLift of `svd_procrustes` ≥ **0.3**;
3. `svd_procrustes` beats the strongest non-learned baseline (identity, cross_lora, random) with a paired-bootstrap 95% CI excluding zero (resampling identical source×target×task cells).

Lift is measured against the **fewshot** base (5 worked training examples in the prompt, shot seed 0). Cells whose mean direct-LoRA lift is below 0.02 are ineligible (ratio unstable).

## Verdict

**NO-GO** — see decision tree (spec §19)

| Criterion | Value | Threshold | Passed |
|---|---|---|---|
| Heterogeneous pairs passing | 1 | ≥ 2 | False |
| Median RecoveredLift | 1.200 | ≥ 0.3 | True |
| Beats strongest baseline | Δ=0.033 [0.033, 0.033] vs `random` (n=1) | CI > 0 | True |

### Ordered pairs (heterogeneous only)

| Source → Target | Eligible tasks | Positive | Passes |
|---|---|---|---|
| tiny_qwen2_b → tiny_llama_a1 | 1 | 1 | True |

### Baselines (mean held-out score over cells shared with the primary method)

| Baseline | Cells | Mean score |
|---|---|---|
| identity | 1 | 0.490 |
| cross_lora | 1 | 0.492 |
| random | 1 | 0.502 |

## Headroom: raw base vs direct LoRA (every base × task cell)

Lift = direct − fewshot base. Flags: **ceiling** = a base scores ≥ 0.9 (no room to measure transfer); **format** = the few-shot prompt alone recovers ≥ 50% of the zero-shot-referenced lift (the adapter mostly teaches output format).

| Base | Task | Metric | Zero-shot | Few-shot | Direct (mean ± sd over seeds) | Lift | Eligible | Flags |
|---|---|---|---|---|---|---|---|---|
| tiny_llama_a1 | T1_sentiment | accuracy | 0.490 | 0.490 | 0.528 ± 0.018 (n=2) | 0.038 | True | – |
| tiny_llama_a1 | T3_paraphrase | accuracy | 0.495 | 0.505 | 0.502 ± 0.004 (n=2) | -0.003 | False | – |
| tiny_qwen2_b | T1_sentiment | accuracy | 0.510 | 0.520 | 0.490 ± 0.000 (n=2) | -0.030 | False | – |
| tiny_qwen2_b | T3_paraphrase | accuracy | 0.495 | 0.495 | 0.502 ± 0.004 (n=2) | 0.007 | False | – |

## Held-out transfer cells (seed-averaged)

RecoveredLift = (S_transfer − S_base) / (S_direct − S_base) with S_base the fewshot base; '–' marks ineligible cells.

| Source → Target | Split | Task | Method | S_transfer | RecoveredLift (mean ± sd) | Δ rel. err | Seeds |
|---|---|---|---|---|---|---|---|
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | cross_lora | 0.495 | – ± – | 1.002 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | identity | 0.495 | – ± – | 1.519 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | procrustes_full | 0.515 | – ± – | 1.480 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | random | 0.495 | – ± – | 1.468 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | svd_linear | 0.495 | – ± – | 1.019 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | svd_procrustes | 0.495 | – ± – | 1.027 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | cross_lora | 0.510 | – ± – | 1.001 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | identity | 0.510 | – ± – | 1.453 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | procrustes_full | 0.510 | – ± – | 1.375 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | random | 0.505 | – ± – | 1.395 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | svd_linear | 0.522 | – ± – | 1.009 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | svd_procrustes | 0.508 | – ± – | 1.019 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | cross_lora | 0.497 | – ± – | 1.002 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | identity | 0.502 | – ± – | 1.379 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | procrustes_full | 0.510 | – ± – | 1.380 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | random | 0.508 | – ± – | 1.396 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | svd_linear | 0.508 | – ± – | 1.021 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | svd_procrustes | 0.500 | – ± – | 1.027 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | cross_lora | 0.492 | 0.100 ± 0.141 | 1.002 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | identity | 0.490 | 0.000 ± 0.000 | 1.414 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | procrustes_full | 0.502 | 0.550 ± 0.919 | 1.486 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | random | 0.502 | 0.250 ± 0.354 | 1.469 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | svd_linear | 0.542 | 1.300 ± 0.990 | 1.019 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | svd_procrustes | 0.535 | 1.200 ± 1.131 | 1.032 | 2 |

## Robustness: alternate prompt format

`v1-alt`: median RecoveredLift of `svd_procrustes` = – over 0 eligible cells.

## Exclusion / failure log

No exclusions recorded.

## Reproduce

```bash
uag analyze --experiment <config for smoke>
```
