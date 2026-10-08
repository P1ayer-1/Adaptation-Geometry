# Is the "Universal Weight Subspace" in LoRA A matrices a shared-seed artefact?

*2026-10-08. Claim tested: Kaushik et al., "The Universal Weight Subspace Hypothesis"
(arXiv 2512.05117), Mistral-7B LoRA analysis (Appendix B.5).*

## Short answer

For the **A matrices, yes.** Of the 500 public adapters analysed, 39% started from one identical
random A, and that group alone produces the low-dimensional subspace. Adapters with their own
seeds show no shared A subspace at all; they look like independent random matrices. The **B
matrices** do share some structure, and it is the same with or without a shared seed, so that
part is real (learned, or set by the base model). It is far weaker than "16 or fewer directions".

## What the paper does

- 500 rank-16 LoRAs for Mistral-7B-Instruct-v0.2 from the Lots-of-LoRAs collection
  (Brüel-Gabrielsson et al. 2024), one per Natural Instructions task, on q/k/v projections.
- For each layer, stack the 500 × 16 rank vectors of A (and separately of B), centre, PCA.
  Headline: most variance in "16 or fewer directions".
- No seed control, no random-matrix baseline.

## Setup

- 500 adapters (`Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task*`, random sample of 903,
  list in `scripts/universal_subspace/ids.txt`), downloaded to the workstation; CPU only.
- Layers 0, 8, 16, 24, 31; q, k, v; A and B. Same stacking, centring and PCA as the paper.
- Seed groups: adapters whose layer-0 q A matrices have cosine > 0.5.

## Results

**1. One seed was reused for 39% of the collection.**
197 of 500 adapters (different tasks) share one starting A; the other 303 each have their own.
Within the group, A matrices are still 62-99% identical (median 90%); between groups, cosine
≈ 0.003. The grouping is identical in every layer and module checked.

**2. The shared A is an untouched PyTorch initialisation.** The group's average A has exactly
the spread of PEFT's default uniform start (std 0.00902 vs 0.00902 expected; kurtosis 1.81 vs
1.80 for a uniform distribution). Individual adapters sit a median 23% away from it.

**3. Matched control: 197 shared-seed adapters vs 197 independent-seed adapters**
(paper's analysis; maximum possible directions 3,152):

| Matrix | Shared seed: variance in top 16 | Shared seed: directions for 90% | Own seeds: variance in top 16 | Own seeds: directions for 90% |
|---|---|---|---|---|
| A (15 layer/module slots) | 0.86-0.90 | 17-80 | 0.02-0.05 | 1,756-1,781 |
| B (15 slots) | 0.12-0.55 | 109-728 | 0.09-0.55 | 113-792 |

- **A:** the "universal subspace" exists only among adapters that share a seed. With independent
  seeds it disappears completely: 2-5% in the top 16, about what independent random matrices give
  (2.6% at N = 100).
- **B:** the shared-seed and own-seed groups agree closely, so this structure is not caused by the
  seed. It is genuinely shared, but needs 100-800 directions for 90% of the variance, not 16.

**4. A fake collection with no training reproduces the full-collection number.** With the same
seed grouping and pure random starts (no training), the paper's analysis on 100 adapters puts
40% of variance in the top 16 directions; the real adapters give 34-36%.

**5. The exact seed was not recovered.** Seeds 0-199 at the first six draw positions, and
15 common seeds (including 42) at any position in the first 67M random draws, did not match;
neither did seed = task number for singleton adapters. Not needed for the conclusion: the
matched control in (3) does not depend on knowing the seed.

## Interpretation

- The paper's LoRA evidence for a universal subspace, as far as it rests on A, is an artefact of
  reused initialisation plus LoRA's A barely moving in training (confirmed independently in our
  dev-scale experiments, where A moved 4-22%).
- Any reported A-side subspace is only as universal as the seeding of the collection. Analyses
  of community LoRAs should group by seed first or remove the initialisation.
- The B-side structure survives the control and is the part worth studying further. It is
  modest (hundreds of directions), and the paper does not report it separately from A.
- Not tested here: the paper's full-model (ViT, ResNet) results, which do not involve LoRA
  initialisation.

## Reproduce

Scripts in `scripts/universal_subspace/` (paths hard-coded to the workstation, `F:/lol`,
`F:/uag`): `dl.py` (download), `subspace.py` (groups, spectra, matched control), `distcheck.py`,
`seedsearch.py` and `streamsearch.py` (seed search). Outputs: `results/universal_subspace/`.
