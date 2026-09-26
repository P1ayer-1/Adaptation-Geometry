# Decision memo — experiment `smoke`

_Generated automatically by `uag analyze` (src/uag/analysis.py) on 2026-09-26 18:39:32 at commit `fd685fa3dae1`. Do not edit by hand; rerun the analysis instead._

> **Pipeline validation only.** This experiment uses randomly initialised tiny bases; its numbers carry no scientific weight.

## Setup

- Bases: `tiny_llama_a1` (tiny_llama, rev `local`), `tiny_qwen2_b` (tiny_qwen2, rev `local`)
- Tasks: `T1_sentiment`, `T3_paraphrase`
- Seeds: [0, 1]; LoRA r=4, alpha=8, modules=['q', 'k', 'v', 'o', 'up', 'down', 'gate']
- Evaluation split: `test` (first 200 examples)
- Held-out splits: `toy_fitT1_predT3` → ['T3_paraphrase']; `toy_fitT3_predT1` → ['T1_sentiment']

## Predeclared gate

Declared 2026-09-26T18:38:04 (sha256 `6491a675990a`), before held-out evaluation.

1. At least **2** heterogeneous ordered base pairs show positive, seed-repeatable held-out transfer on a majority of eligible tasks;
2. median RecoveredLift of `svd_procrustes` ≥ **0.3**;
3. `svd_procrustes` beats the strongest non-learned baseline (identity, cross_lora, random) with a paired-bootstrap 95% CI excluding zero (resampling identical source×target×task cells).

Cells whose mean direct-LoRA lift is below 0.02 are ineligible (ratio unstable).

## Verdict

**NO-GO** — see decision tree (spec §19)

| Criterion | Value | Threshold | Passed |
|---|---|---|---|
| Heterogeneous pairs passing | 0 | ≥ 2 | False |
| Median RecoveredLift | – | ≥ 0.3 | False |
| Beats strongest baseline | – | CI > 0 | False |

### Ordered pairs (heterogeneous only)

| Source → Target | Eligible tasks | Positive | Passes |
|---|---|---|---|
| (no heterogeneous pairs with eligible cells) | | | |

### Baselines (mean held-out score over cells shared with the primary method)

| Baseline | Cells | Mean score |
|---|---|---|
| identity | 0 | – |
| cross_lora | 0 | – |
| random | 0 | – |

## Headroom: raw base vs direct LoRA (every base × task cell)

| Base | Task | Metric | Base | Direct (mean ± sd over seeds) | Lift | Eligible |
|---|---|---|---|---|---|---|
| tiny_llama_a1 | T1_sentiment | accuracy | 0.490 | 0.505 ± 0.014 (n=2) | 0.015 | False |
| tiny_llama_a1 | T3_paraphrase | accuracy | 0.495 | 0.502 ± 0.004 (n=2) | 0.007 | False |
| tiny_qwen2_b | T1_sentiment | accuracy | 0.510 | 0.490 ± 0.000 (n=2) | -0.020 | False |
| tiny_qwen2_b | T3_paraphrase | accuracy | 0.495 | 0.502 ± 0.004 (n=2) | 0.007 | False |

## Held-out transfer cells (seed-averaged)

RecoveredLift = (S_transfer − S_base) / (S_direct − S_base); '–' marks ineligible cells.

| Source → Target | Split | Task | Method | S_transfer | RecoveredLift (mean ± sd) | Δ rel. err | Seeds |
|---|---|---|---|---|---|---|---|
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | cross_lora | 0.495 | – ± – | 1.002 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | identity | 0.495 | – ± – | 1.519 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | procrustes_full | 0.495 | – ± – | 1.451 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | random | 0.495 | – ± – | 1.463 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | svd_linear | 0.492 | – ± – | 1.013 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT1_predT3 | T3_paraphrase | svd_procrustes | 0.495 | – ± – | 1.018 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | cross_lora | 0.510 | – ± – | 1.001 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | identity | 0.510 | – ± – | 1.482 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | procrustes_full | 0.510 | – ± – | 1.402 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | random | 0.508 | – ± – | 1.399 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | svd_linear | 0.522 | – ± – | 1.007 | 2 |
| tiny_llama_a1 → tiny_qwen2_b | toy_fitT3_predT1 | T1_sentiment | svd_procrustes | 0.508 | – ± – | 1.012 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | cross_lora | 0.497 | – ± – | 1.002 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | identity | 0.502 | – ± – | 1.379 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | procrustes_full | 0.495 | – ± – | 1.423 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | random | 0.508 | – ± – | 1.401 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | svd_linear | 0.508 | – ± – | 1.016 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT1_predT3 | T3_paraphrase | svd_procrustes | 0.518 | – ± – | 1.018 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | cross_lora | 0.482 | – ± – | 1.002 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | identity | 0.482 | – ± – | 1.381 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | procrustes_full | 0.497 | – ± – | 1.444 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | random | 0.500 | – ± – | 1.464 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | svd_linear | 0.537 | – ± – | 1.014 | 2 |
| tiny_qwen2_b → tiny_llama_a1 | toy_fitT3_predT1 | T1_sentiment | svd_procrustes | 0.532 | – ± – | 1.022 | 2 |

## Robustness: alternate prompt format

`v1-alt`: median RecoveredLift of `svd_procrustes` = – over 0 eligible cells.

## Exclusion / failure log

No exclusions recorded.

## Reproduce

```bash
uag analyze --experiment <config for smoke>
```
