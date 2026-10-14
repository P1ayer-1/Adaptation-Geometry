# Is the "Universal Weight Subspace" in LoRA A matrices a shared-seed artefact?

*2026-10-08. Claim tested: Kaushik et al., "The Universal Weight Subspace Hypothesis"
(arXiv 2512.05117), Mistral-7B LoRA analysis (Appendix B.5).*

## Short answer

**Mostly, yes.** Of the 500 public adapters analysed, 39% started from one identical random A, and
that group alone produces the low-dimensional A subspace. Adapters with their own seeds show no
shared A subspace at all. The B matrices and the full updates (B·A) look more concentrated than
random, but almost all of that comes from uneven adapter strengths (a few strong directions per
adapter, a few strong adapters), not from shared directions: a null with the same strengths and
random directions reproduces it. What remains is a small shared output-side structure, clear
only in layer 0, where it lines up with the base model's own dominant directions. The seed also
drives compressibility: the joint-basis compression used for this collection compresses
shared-seed adapters far better than own-seed ones.

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
  seed. Most of it is explained by uneven strengths, not shared directions; see result 6.

**4. A fake collection with no training reproduces the full-collection number.** With the same
seed grouping and pure random starts (no training), the paper's analysis on 100 adapters puts
40% of variance in the top 16 directions; the real adapters give 34-36%.

**5. The exact seed was not recovered.** Seeds 0-199 at the first six draw positions, and
15 common seeds (including 42) at any position in the first 67M random draws, did not match;
neither did seed = task number for singleton adapters. Not needed for the conclusion: the
matched control in (3) does not depend on knowing the seed.

**6. Full update (B·A), gauge-free, with a strength-matched null.** Each adapter is rewritten as
its compact SVD, ΔW = Σ s_j u_j v_jᵀ, and the paper's stacking and PCA are applied to the
input-side vectors s_j v_j and output-side vectors s_j u_j. The null keeps every adapter's
singular values s_j but gives it random orthonormal directions.

| Top-16 variance share | Shared seed | Own seeds | Null (own seeds' strengths, random directions) |
|---|---|---|---|
| Input side (6 slots) | 0.56-0.67 | 0.13-0.38 | — |
| Input side, own seeds vs null (15 slots) | | 0.10-0.38 | 0.10-0.34 (gap ≤ 0.04, except L0 v 0.08) |
| Output side, own seeds vs null (15 slots) | | 0.14-0.59 | 0.10-0.34 (gap 0.02-0.10; L0 q 0.20, L0 k 0.34) |

- Input side: the shared seed still shows up after removing the A/B split (0.56-0.67 against
  0.13-0.38), so result 3 is not an artefact of looking at A alone. Without a shared seed, the
  input side is at the null in every layer except a little in layer 0.
- Output side: a small excess over the null in most slots, a large one only in layer 0.

**7. The remaining output structure is partly the base model.** Mean overlap (cos² of principal
angles) between the own-seed adapters' top-16 shared output subspace and the base weight's
top-16 left singular vectors: 0.06-0.19 in layer 0 (chance 0.004-0.016, so 10-50× chance), and
0.009-0.032 elsewhere (about 1-3× chance). Share of total update energy inside the base weight's
top-128 output directions: 0.17-0.67 in layer 0 against 0.03-0.13 chance; at chance for k/v in
later layers, ~1.5× for q. On the input side, overlap with the base weight's top directions is
0.08-0.12 in layer 0 and near chance later; overlap with the dominant directions of the actual
layer-0 inputs (normalised embeddings) is low (0.005-0.017).

**8. Compression depends on the seed.** Compress then Serve (Brüel-Gabrielsson et al. 2024, the
same group that trained these adapters) compresses LoRAs into shared bases U, V with a small
per-adapter core, and reports that clustering the LoRAs "significantly aids performance" for
n ≥ 100, with relative error below 0.6 as the safe threshold. Their training section gives no
seeds. We ran JD-Full (10 alternating iterations, unit-norm updates) on 100 adapters per set:

| Relative error, rank 16 | Shared-seed set | Own-seed set | Mixed 50/50: shared half | Mixed: own-seed half |
|---|---|---|---|---|
| 6 slots | 0.62-0.85 | 0.86-0.90 | 0.62-0.78 | 0.89-1.00 |

At rank 32 the gap persists (mixed: 0.51-0.73 vs 0.68-0.86); at rank 64 it mostly closes. In a
mixed collection the shared basis is spent on the shared-seed adapters. Their clustering result
may partly rediscover seed groups; the collection's seeding should be checked before reading it
as task structure. (Our JD is a simplified re-implementation: no clustering, low-rank SVD updates.)

## Interpretation

- The paper's LoRA evidence for a universal subspace is, on the input side, an artefact of reused
  initialisation plus LoRA's A barely moving in training (confirmed independently in our
  dev-scale experiments, where A moved 4-22%). It persists in the gauge-free full update.
- Without the shared seed, almost all of the remaining concentration is explained by uneven
  adapter strengths. A real shared structure is left on the output side, small except in layer 0,
  where it is tied to the base model's dominant directions: a property of Mistral more than of
  adaptation.
- Analyses of LoRA collections should group by seed (or remove the initialisation) and compare
  against a strength-matched null, not against equal-weight random matrices.
- The same applies to methods that exploit shared structure, such as joint-basis compression.
- Not tested here: the paper's full-model (ViT, ResNet) results, which do not involve LoRA
  initialisation.

## Reproduce

Scripts in `scripts/universal_subspace/` (paths hard-coded to the workstation, `F:/lol`,
`F:/uag`): `dl.py` (download), `subspace.py` (groups, spectra, matched control), `distcheck.py`,
`seedsearch.py` and `streamsearch.py` (seed search), `robust.py` (full update, compression),
`dlbase.py` and `basecheck.py` (null and base-model comparison). Outputs: `results/universal_subspace/`.
