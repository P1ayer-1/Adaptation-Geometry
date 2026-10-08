# A Shared Seed, Not a Universal Subspace: Revisiting the LoRA Evidence for the Universal Weight Subspace Hypothesis

*Draft, 2026-10-08. [Author]. Items marked **[TODO]** must be resolved before posting.*

## Abstract

Kaushik et al. (2025) report that 500 LoRA adapters for Mistral-7B, each trained on a different
task, share a low-dimensional "universal" subspace, with most information in 16 or fewer
directions per layer. LoRA initialises its down-projection A at random and its up-projection B
at zero, and A moves little during training. We therefore checked whether the shared subspace
comes from shared initialisation. In 500 adapters from the same public collection, 39% (197)
start from one identical random A, still 90% identical (median cosine) after training. The
remaining 303 each have their own seed. With the paper's analysis, the shared-seed adapters put
86-90% of A's variance in 16 directions; an equal number of own-seed adapters put 2-5%, the
level of independent random matrices. A simulated collection with the same seed grouping and no
training at all reproduces the full collection's spectrum. Representing each update gauge-free
(ΔW = BA) does not remove the effect. Comparing own-seed adapters with a null that keeps each
adapter's singular values but randomises its directions, almost all the remaining concentration
is explained by uneven adapter strengths. A small shared output-side structure remains, large
only in the first layer, where it aligns with the base model's dominant weight directions. The
seed also governs compressibility: joint-basis compression of the kind used to serve this
collection fits shared-seed adapters much better than own-seed ones. We recommend reporting
seeds, grouping by seed, and using strength-matched nulls in analyses of adapter collections.

## 1. Introduction

The Universal Weight Subspace Hypothesis (Kaushik et al., 2025, arXiv 2512.05117) proposes that
neural networks converge to shared spectral subspaces "regardless of initialization, task, or
domain" **[TODO: verify quote against the paper]**. One line of evidence is a collection of 500
LoRA adapters (Hu et al., 2021) for Mistral-7B-Instruct-v0.2, each trained on a Natural
Instructions task (Wang et al., 2022) and released by Brüel-Gabrielsson et al. (2024). For each
layer, the paper stacks the adapters' A matrices (and separately their B matrices), centres them
and runs a principal component analysis. It reports that most of the variance lies in 16 or
fewer directions (Appendix B.5). It reports no seed information and no random-matrix baseline
for this analysis.

Two facts about LoRA make initialisation a candidate explanation. First, PEFT initialises A with
a Kaiming-uniform draw from the global random number generator, so adapters trained with the
same seed and configuration start from an identical A. Second, A moves little in training. In
our own controlled experiments on Qwen2.5-0.5B and Llama-3.2-1B, A moved 4-22% from its
initialisation across 40 runs. Under a shared seed, unrelated tasks showed 97-99% overlap of
their input subspaces, coming entirely from the shared start. If many adapters in a collection
share a seed, the stacked A matrices contain the same r = 16 random vectors many times over.
PCA will then find about 16 dominant directions, whatever the tasks.

We test this on the paper's collection. Our contributions:

1. We show that 39% of a 500-adapter sample shares one random initialisation, and that A is
   still close to it after training (Section 3.1).
2. With a matched comparison of shared-seed and own-seed adapters, we show the A-side subspace
   exists only under the shared seed (Section 3.2), and that a no-training simulation reproduces
   the full-collection spectrum (Section 3.3).
3. Using a gauge-free representation and a strength-matched null, we show what structure remains
   once the seed is removed: little, mostly in layer 0 and tied to the base model
   (Sections 3.4-3.5).
4. We show the same seed effect in joint-basis compression of this collection (Section 3.6).

We do not test the paper's results on fully trained models (ViTs, ResNets), which do not involve
LoRA initialisation.

## 2. Data and methods

**Adapters.** The Lots-of-LoRAs collection on Hugging Face has 903 rank-16 adapters for
Mistral-7B-Instruct-v0.2 (`Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task*`). We use a
random sample of 500, matching the paper's count. Each adapts q, k and v projections in all 32
layers (α = 32, `init_lora_weights: true`). The paper's exact 500 are not listed
**[TODO: check the paper's appendix or code for the adapter list]**. We analyse layers 0, 8, 16,
24 and 31.

**Paper-style spectrum.** For each layer and module, we stack the 16 rows of every adapter's A
(N·16 × 4096) or the 16 columns of B (N·16 × d_out), centre feature-wise, and compute the
eigenvalues of the covariance. We report the variance share of the top 16 components and the
number of components needed for 90% of the variance.

**Seed groups.** We call two adapters seed-sharing if their layer-0 q_proj A matrices have cosine
similarity above 0.5, and take connected components.

**Gauge-free representation.** ΔW = BA is invariant to A → MA, B → BM⁻¹, but A and B separately
are not. We therefore also compute each adapter's compact SVD, ΔW = Σⱼ sⱼ uⱼ vⱼᵀ (via QR of B and
Aᵀ), and apply the same spectrum analysis to the input-side vectors sⱼvⱼ and output-side vectors
sⱼuⱼ.

**Strength-matched null.** Adapters differ in norm, and each adapter's singular values are
uneven. Both concentrate a stacked spectrum even when directions are unrelated. The null keeps
every adapter's singular values sⱼ and replaces its singular vectors with random orthonormal ones.

**Compute.** All analyses run on a CPU (12 threads, 16 GB RAM) in a few hours in total. Code and outputs
are in the accompanying repository (Section 6).

## 3. Results

### 3.1 One seed was reused for 39% of the collection

197 of the 500 adapters share one starting A; the other 303 each have their own. Within the
shared group, A matrices for different tasks have cosine of at least 0.62 (median 0.90). Between groups
the cosine is about 0.003 (maximum 0.11). The grouping is identical in every layer and module we
checked. The tasks in the shared group are not a contiguous range of task numbers.

The group's average A is statistically an untouched PyTorch initialisation. Its element standard
deviation is 0.00902, against 0.00902 expected for U(−1/64, 1/64). Its kurtosis is 1.81, against
1.80 for a uniform distribution and 3.0 for a Gaussian. Individual adapters lie a median 23%
(10th-90th percentile 7-62%) of the init's norm away from it.

We did not recover the seed. Seed 42 and 14 other common seeds did not match within the first 67 million draws of
PyTorch's CPU generator, and seed = task number did not match for four own-seed adapters
(first 17 million draws). Seeds 0-199 did not match at the
first six draw positions. The random stream was probably advanced by other code before the
adapters were created, or the adapters were created on a GPU. None of the analyses below need
the seed.

### 3.2 Matched control: the A subspace exists only under a shared seed

We compare the 197 shared-seed adapters with 197 randomly chosen own-seed adapters (maximum
3,152 components).

| Paper-style spectrum | Shared seed: top-16 share | Shared seed: comps. for 90% | Own seeds: top-16 share | Own seeds: comps. for 90% |
|---|---|---|---|---|
| A, 15 layer-module slots | 0.86-0.90 | 17-80 | 0.02-0.05 | 1,756-1,781 |
| B, 15 slots | 0.12-0.55 | 109-728 | 0.09-0.55 | 113-792 |

For A, own-seed adapters are indistinguishable from independent random matrices (top-16 share
0.011 for 500 random draws). Subtracting the shared group's mean A (an estimate of the initialisation) lowers the
shared group's top-16 share from 0.86-0.90 to 0.10-0.31. For
B, the two groups agree, so B's concentration is not a seed effect. Section 3.4 shows it is
mostly not shared directions either.

### 3.3 A collection with no training reproduces the spectrum

On the full 500, the paper-style A spectrum puts 33-35% of variance in the top 16 components,
followed by a sharp drop. A simulated collection with the same seed grouping, pure Kaiming
initialisations and no training gives 39%. Independent initialisations give 1.1%. The location
of the drop, at exactly the LoRA rank, is what a shared initialisation predicts: 197 copies of
the same 16 vectors contribute 16 large eigenvalues. **[TODO: add Figure 1: cumulative variance
curves for the real collection, the no-training simulation, own-seed adapters and independent
random matrices.]**

### 3.4 Gauge-free updates and a strength-matched null

| Top-16 share, gauge-free ΔW | Shared seed | Own seeds | Null for own seeds |
|---|---|---|---|
| Input side (sⱼvⱼ), 6 slots | 0.56-0.67 | 0.13-0.38 | — |
| Input side, 15 slots | | 0.10-0.38 | 0.10-0.34 |
| Output side (sⱼuⱼ), 15 slots | | 0.14-0.59 | 0.10-0.34 |

On the input side, the shared seed still dominates in the gauge-free representation, so the
effect is not an artefact of analysing A alone. Without a shared seed, the input side matches
the strength-matched null to within 0.04 in all slots but one (layer 0 v: 0.20 vs 0.12). On the
output side, own-seed adapters exceed the null by 0.02-0.10 in most slots. The excess is large
only in layer 0 (q: 0.54 vs 0.34; k: 0.59 vs 0.25).

Equal-weight random baselines are therefore misleading for adapter collections. Uneven adapter
norms and uneven singular values alone produce a top-16 share of 0.10-0.34 in this collection.

### 3.5 What remains is tied to the base model

We compare the own-seed adapters' top-16 shared output subspace with the top-16 left singular
vectors of the corresponding base weight (mean cos² of principal angles). The overlap is
0.06-0.19 in layer 0, against chance of 0.004-0.016 (10-50×), and 0.009-0.032 in later layers
(about 1-3×). In layer 0, 17-67% of the adapters' total update energy lies in the base weight's
top-128 output directions, against 3-13% by chance. In later layers it is near chance for k and
v and about 1.5× chance for q. The residual shared structure is therefore largely a property of
the base model's early layers rather than a task-general adaptation subspace.

### 3.6 The seed also governs compressibility

Brüel-Gabrielsson et al. (2024), who trained this collection, serve many LoRAs by approximating
each update as U Σᵢ Vᵀ with bases U, V shared across adapters. They report that clustering
adapters "significantly aids performance" for n ≥ 100, and use a relative reconstruction error
below 0.6 as a safe threshold **[TODO: verify quotes]**. We re-implemented their JD-Full variant
(unit-norm updates, 10 alternating iterations, no clustering) on 100-adapter sets.

| Relative error | Shared-seed set | Own-seed set | Mixed 50/50: shared half | Mixed: own-seed half |
|---|---|---|---|---|
| Rank 16 (6 slots) | 0.62-0.85 | 0.86-0.90 | 0.62-0.78 | 0.89-1.00 |
| Rank 32 | 0.43-0.74 | 0.72-0.82 | 0.51-0.73 | 0.68-0.86 |
| Rank 64 | 0.25-0.58 | 0.50-0.68 | 0.35-0.60 | 0.37-0.67 |

At low rank, a shared basis fitted to a mixed collection is spent mainly on the shared-seed
adapters. Clusters found in such a collection may partly be seed groups rather than task groups.
Our implementation is simplified, and we did not test downstream task performance.

## 4. Discussion

**What the LoRA evidence supports.** The A-side "universal subspace" in this collection is a
reused random initialisation that training barely changes. In the gauge-free update the
input-side effect remains a seed effect. After removing the seed and accounting for uneven
strengths, a modest shared output-side structure remains. It is substantial only in the first
layer, where it follows the base model's dominant directions. This is far from "most
information in 16 or fewer directions", and it is better described as a property of the base
model than as a universal adaptation subspace.

**Recommendations for analyses of adapter collections.**

1. Report how initialisations were seeded; check for shared initialisations directly (pairwise
   cosine of A is enough).
2. Analyse seed groups separately, or subtract the initialisation, before pooling.
3. Use gauge-free representations of ΔW rather than A and B separately.
4. Compare against a null that keeps each adapter's singular values, not against equal-weight
   random matrices.
5. Apply the same checks to methods that exploit shared structure, such as joint-basis
   compression and clustering, and to continual-learning methods that use angles between LoRA
   subspaces.

**Limitations.** We analysed one collection, five of 32 layers and a random 500 of 903 adapters
rather than the paper's exact set. We did not recover the shared seed. Our compression
experiment is a simplified re-implementation without clustering or downstream evaluation. The
base-model comparison uses weight singular vectors as a proxy for the directions a layer favours;
activation statistics would be more direct. Our claims concern the LoRA evidence only, not the
paper's experiments on fully trained models.

**A constructive direction.** Shared structure in adapters is only interpretable if the update
shape is determined by the task rather than the seed. Initialising A from the task's gradient
(LoRA-GA, Wang et al., 2024; LoRA-One, Zhang et al., 2025) or using preconditioned updates
(Zhang & Pilanci, 2024) are candidate ways to obtain seed-invariant adapters. Whether they do so
is an open, testable question. **[TODO: cite or summarise the follow-up experiment if run.]**

## 5. Related work

**[TODO: short paragraph.]** LoRA initialisation and its effects (PiSSA, LoRA-GA, LoRA-One);
asymmetry between A and B (Zhu et al., 2024); recycling adapters into shared bases (EigenLoRAx,
2025; Compress then Serve, 2024); shared LoRA subspaces for continual learning (O-LoRA; "Shared
LoRA Subspaces for almost Strict Continual Learning", arXiv 2602.06043).

## 6. Reproducibility

Code: `scripts/universal_subspace/` (download, seed grouping and spectra, seed search,
gauge-free analysis, compression, null and base-model comparison). Outputs:
`results/universal_subspace/`. Adapter list: `scripts/universal_subspace/ids.txt` (the first 500
lines are the sample). **[TODO: public repository link; remove hard-coded paths.]**

## References

**[TODO: complete and verify all entries.]**

- Brüel-Gabrielsson, R., Zhu, J., Bhardwaj, O., Choshen, L., Greenewald, K., Yurochkin, M., Solomon, J. (2024). Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead. arXiv 2407.00066.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv 2106.09685.
- Kaushik, et al. (2025). The Universal Weight Subspace Hypothesis. arXiv 2512.05117.
- Wang, S., et al. (2024). LoRA-GA: Low-Rank Adaptation with Gradient Approximation. arXiv 2407.05000.
- Wang, Y., et al. (2022). Super-NaturalInstructions. EMNLP 2022.
- Zhang, F., Pilanci, M. (2024). Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models. arXiv 2402.02347.
- Zhang, Y., et al. (2025). LoRA-One. arXiv 2502.01235.
