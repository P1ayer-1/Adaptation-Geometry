# Shared Initialisation Confounds the LoRA Evidence for Universal Weight Subspaces

*Draft v3, 2026-10-10. [Author]. Revised after three external reviews (see the revision logs at the
end). Items marked **[TODO]** must be resolved before posting.*

## Abstract

Kaushik et al. (arXiv 2512.05117, v3) report that about 500 LoRA adapters for Mistral-7B, each
trained on a different task, share a low-dimensional "universal" subspace, with "most information
concentrated in 16 or fewer directions across all layers". LoRA initialises its down-projection A
at random and its up-projection B at zero, and A moves little in training. We therefore asked how
much of this structure comes from shared initialisation. On the adapters listed in the paper's Table 10 (497 inferred to have been used for fitting;
including all 502 changes nothing material), 255 of 497 (51%) form one group whose A matrices are nearly identical across
different tasks (median pairwise cosine 0.90), consistent with a reused random initialisation.
Across the whole public collection the fraction is 38%, so the paper's set is enriched for it. In
the paper's analysis, this group puts 86-90% of A's variance in 16 directions; the 242 adapters
outside it put 2-5%, close to independent random matrices at the same sample size (1.6%). A
simulated collection with the same grouping and no training at all shows the same rank-16 knee.
The effect survives a factorisation-invariant analysis of ΔW = BA on the input side. The output
side behaves differently: it shows shared structure that is present in both inferred groups,
exceeds a null that preserves each adapter's singular values in every layer we examined, and
outside the first layer is little reduced by removing the base model's dominant weight directions. In a controlled experiment on Qwen2.5-0.5B, the same ten tasks show the rank-16 A subspace when they share an initialisation and the random level when they do not; a subspace fitted on other tasks does not reconstruct a held-out task, and projecting one side at a time shows that within a shared initialisation the output side is what fails. We conclude that the input-side
(A) evidence for a universal LoRA subspace is largely an initialisation artefact, while a weaker,
genuinely shared output-side structure deserves separate study. We recommend that analyses of
adapter collections report seeds, group by inferred initialisation, use factorisation-invariant
statistics and compare against strength-matched nulls.

## 1. Introduction

The Universal Weight Subspace Hypothesis (Kaushik et al., arXiv 2512.05117; we review v3, dated
2026-10-05) proposes that neural networks converge to shared spectral subspaces "regardless of
initialization, task, or domain". One line of evidence is a collection of LoRA adapters (Hu et al.,
2021) for Mistral-7B-Instruct-v0.2, each trained on a Natural Instructions task (Wang et al., 2022)
and released by Brüel-Gabrielsson et al. (2024). For each layer the paper stacks the adapters'
rank vectors into an (N·r) × d matrix, separately for A and B, centres it, and runs a principal
component analysis (Appendix B.5) **[TODO: confirm the stacking description in v3]**. The paper
acknowledges that "the LoRA factorization is not unique" and that its analysis "characterizes
recurring structure in the learned and stored LoRA parameterization ... rather than a
gauge-invariant property of the equivalent full update". It reports no seed information and no
random-matrix baseline for this analysis.

Two facts make initialisation a candidate explanation. First, PEFT initialises A with a
Kaiming-uniform draw from the global random generator, so adapters trained with the same seed and
configuration start from an identical A. Second, A moves little in training. In our own controlled
runs on Qwen2.5-0.5B and Llama-3.2-1B (40 runs; [`dev_scale_report.md`](dev_scale_report.md)),
‖A − A₀‖_F / ‖A₀‖_F was 0.04-0.22, and under a shared seed the input subspaces of unrelated tasks
overlapped by 97-99%. If many adapters in a collection share an initialisation, the stacked A
matrix contains the same r = 16 random vectors many times, and PCA finds about 16 dominant
directions whatever the tasks.

Contributions:

1. On the paper's adapter list (the inferred fitting set, Section 2), we identify one large group of adapters with a common,
   initialisation-like A (51%), robust to the grouping threshold and identical in all layers
   examined (Section 3.1).
2. A matched comparison, size-matched random nulls and a no-training simulation show that the
   rank-16 A spectrum comes from this group (Section 3.2, Figure 1).
3. A crossed task × initialisation experiment on Qwen2.5-0.5B separates seed from task directly,
   and one-sided projections locate where a fitted subspace fails to carry a new task (Section 3.3).
4. A factorisation-invariant analysis with strength-matched nulls separates what the shared
   initialisation explains (input side) from genuinely shared structure (output side), and tests
   how much of the latter is the base model's dominant directions (Sections 3.4-3.5).
5. The same grouping affects joint-basis compression of this collection (Section 3.6).

We do not test the paper's results on fully trained models (ViTs, ResNets), which do not involve
LoRA initialisation.

## 2. Setup

**Adapters.** Table 10 of v3 lists 502 adapters `Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-taskNNN`
(all present in the public collection of 904 rank-16 adapters). Five are coloured as
out-of-distribution evaluation models. We infer from the colouring that the other 497 were used to
fit the subspace and call them the *inferred fitting set*; "the paper's set" below means this set.
The split is not confirmed by the authors, so Section 3.1 reports the sensitivity to including all
502. Each adapter
adapts the q, k and v projections of all 32 layers with rank r = 16 and scaling α/r = 2 (constant,
so it cancels in every normalised statistic below). A is r × d_in (d_in = 4,096); B is
d_out × r, with d_out = 4,096 for q and 1,024 for k and v (grouped-query attention). A **slot** is
one (layer, projection) pair; we analyse 15 slots: layers 0, 8, 16, 24, 31 × q, k, v.

**Paper-style spectrum.** For a set of N adapters, stack the r rows of each A (or the r columns of
each B) into an (N·r) × d matrix, centre feature-wise and compute eigenvalues. **Top-16 share** is
the fraction of variance in the 16 largest components; **k90** is the number of components needed
for 90% of the variance.

**Factorisation-invariant spectra.** Writing ΔW_i = B_i A_i, we use the uncentred operators
C_in = Σᵢ ΔW_iᵀ ΔW_i and C_out = Σᵢ ΔW_i ΔW_iᵀ, whose spectra do not depend on how ΔW_i is split
into factors (computed from each adapter's compact SVD via QR of B and Aᵀ). We report their top-16
share. These are one-sided measures of update energy, not PCA over vectorised updates.

**Nulls.** (i) *Size-matched independent initialisations*: N fresh Kaiming-uniform A matrices,
10 repeats. (ii) *Strength-matched*: every adapter keeps its singular values and gets random
orthonormal singular vectors, 10 repeats; this tests orientation given strength. (iii) *Unit-norm*
versions of the real data and null (every ΔW_i scaled to ‖ΔW_i‖_F = 1), which remove differences
in adapter norm and keep only within-adapter anisotropy.

**Inferred groups.** Adapters are linked when the cosine similarity of their layer-0 q_proj A
matrices exceeds 0.5; groups are connected components. We call the large component the *inferred
shared-initialisation group* and the rest *singletons*. These are inferences from A, not recovered
training histories.

## 3. Results

### 3.1 One large group shares an initialisation-like A

On the paper's 497 adapters the grouping yields one group of 255 (51%) and 242 singletons; no
other multi-member groups. Within the group, pairwise cosine of layer-0 q_proj A has minimum 0.55,
10th percentile 0.78 and median 0.90. Between groups, 98% of cosines lie within ±0.010 and the
largest is 0.044. The partition is identical for every threshold from 0.15 to 0.6, and the same
partition is obtained independently from each of the 15 slots.

*Sensitivity to the inferred split.* Including the five out-of-distribution adapters (all 502)
leaves the partition unchanged: none of the five joins the group, which stays at 255 members, and
the other 247 adapters are singletons. The full-set top-16 share of A changes by −0.004 to −0.005 in
every slot (0.43-0.45; k90 1,632-1,759).

The group's mean A behaves like an initialisation plus a small averaged change. Its kurtosis is
1.80-1.84 across slots (1.80 for a uniform distribution, 3.0 for a Gaussian); its element standard
deviation is 0.00901-0.00908 (0.00902 for PEFT's U(−1/64, 1/64)); its largest entries reach
0.016-0.020, slightly beyond the uniform bound of 0.0156, as expected if the mean includes some
learned change. Individual adapters lie a median 0.22-0.32 (10th-90th percentile 0.06-0.65) of
the mean's norm away from it. These are distances from the trained group mean, not from the
initialisation, and they bound neither an individual adapter's movement nor the median movement.
Only an aggregate statement holds: since Σᵢ‖Aᵢ − m‖² ≤ Σᵢ‖Aᵢ − c‖² for the mean m and any common
point c, the root-mean-square distance from the mean, 0.33-0.39 of the mean's norm across slots,
cannot exceed the root-mean-square distance from the shared initialisation, measured in the same
units.

**Collection-wide.** Over all 904 rank-16 adapters, 342 (38%) belong to the group. The fraction
varies with task number (25-51% across ranges). The paper's set (51%) is enriched relative to a
random 500 (39%), the lowest-numbered 500 (44%) and the whole collection (38%), and the top-16
share of layer-0 q_proj A tracks the group fraction across these sets (0.45, 0.34, 0.38, 0.32).

The model cards and `adapter_config.json` files contain no seed and no evaluation results. Seed 42
and 14 other common seeds did not match the group mean within the first 67 million draws of
PyTorch's CPU generator, and seeds 0-199 did not match at the first six draw positions; the
generator may have been advanced by other code, or the adapters created on a GPU. Nothing below
depends on the seed.

### 3.2 The rank-16 A spectrum comes from the group

| Paper-style A spectrum (15 slots) | Top-16 share | k90 |
|---|---|---|
| Paper's set (497 adapters) | 0.44-0.46 | 1,610-1,738 |
| Same grouping, pure initialisations, no training | 0.51 | — |
| Same grouping, initialisations + random movement of the measured size | 0.46-0.48 | — |
| Shared-initialisation group (242 of 255) | 0.86-0.90 | 16-90 |
| Singletons (242) | 0.021-0.052 | 1,980-2,003 |
| Independent random initialisations (242, 10 repeats) | 0.016 (sd < 0.001) | — |

(Rank caps: 4,096 for the full set; 3,871 for 242 adapters after centring.)

The paper's set shows a sharp knee at exactly the LoRA rank (Figure 1). A collection with the same
grouping and no training at all reproduces the knee, and adding random movement of the measured
size brings the top-16 share to within 0.02-0.04 of the real value. Under row stacking, the top-16
share of such a collection is roughly the shared fraction times the share of each group member's
variance carried by the common A, which is why it tracks the group fraction (Section 3.1).
Singletons are 1.3-3.3 times the size-matched random level, highest in layer 0 (0.036-0.052), so
they are close to, but not exactly, random. The rank-16 concentration is absent among them.

For B, the group and the singletons agree to within 0.015 in every slot (top-16 share 0.11-0.56),
so B's spectrum does not depend on group membership.

![Figure 1](figures/figure1_spectrum.png)

*Figure 1. Cumulative variance of the paper-style A spectrum, layer 16 q_proj. The paper's set and
a no-training simulation with the same grouping both jump at the LoRA rank; adapters outside the
group are close to independent random matrices.*

### 3.3 Crossed task × initialisation experiment

To separate initialisation from task directly, we trained the same 10 tasks (our synthetic panel:
hidden-rule sentiment, NLI, paraphrase, JSON extraction, concise and verbose rewriting, Python,
arithmetic, clinical extraction, formatting) on Qwen2.5-0.5B, rank 16 on all seven projection
types (168 modules), with identical data, optimiser and early stopping, under two regimes:
**shared**, one random A per seed used by every task (two seeds, so two shared initialisations),
and **independent**, a separate random A per task and seed. A hardware check retrained four
independent runs on a different GPU (RTX 3080 vs A100) from the same initialisation: ΔW cosine
0.88-0.94 and A cosine ≥ 0.997, so training is reproducible across machines.

| Paper-style A spectrum, 10 adapters per group | Top-16 share (median over 168 modules) |
|---|---|
| Shared initialisation (each of the two groups) | 0.99 |
| Both shared groups pooled (20 adapters) | 0.59 |
| Independent initialisations | 0.18 |
| Independent random matrices, same N | 0.18 |

With everything else fixed, the rank-16 subspace appears when tasks share an initialisation,
splits into two subspaces when two initialisations are pooled, and is at the random level when
initialisations are independent.

We then fitted rank-16 input and output bases (top singular vectors of the stacked A rows and B
columns) on nine tasks and projected the held-out task's adapter onto them
(ΔW′ = U Uᵀ B A V Vᵀ):

| Basis fitted on | Share of held-out ‖ΔW‖²_F captured (median, range over 10 tasks) |
|---|---|
| Other tasks, same initialisation | 0.070 (0.029-0.115) |
| Other tasks, other initialisation | 0.001 (0.001-0.002) |
| All tasks *including the same task*, other initialisation | 0.006 (0.002-0.011) |
| Other tasks, independent initialisations | 0.002 (0.001-0.004) |

A two-sided projection cannot say which side loses the update, so we also projected one side at a
time: input only (ΔW′ = B A V Vᵀ, keeping the adapter's own output directions) and output only
(ΔW′ = U Uᵀ B A, keeping its own input directions).

| Basis fitted on other tasks | Input side only | Output side only | Both sides |
|---|---|---|---|
| Same initialisation | 0.945 (0.810-0.993) | 0.074 (0.034-0.127) | 0.070 (0.029-0.115) |
| Other initialisation | 0.017 (0.016-0.018) | 0.072 (0.033-0.120) | 0.001 (0.001-0.002) |
| Independent initialisations | 0.019 (0.018-0.021) | 0.085 (0.034-0.130) | 0.002 (0.001-0.004) |

(Share of the held-out ‖ΔW‖²_F captured; median and range over 10 tasks. Rows 1-2 hold out an
adapter from one shared-initialisation group; row 3 holds out an independently initialised one.)

| Task score (test, n = 200) | Direct adapter | Projected, same init | Projected, other init | Projected, independent | Base model |
|---|---|---|---|---|---|
| Paraphrase | 0.99-1.00 | 0.71 | 0.65 | 0.65 | 0.65 |
| JSON extraction | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Arithmetic | 0.90 | 0.01 | 0.00 | 0.00 | 0.00 |
| Clinical extraction | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |

| Task score, one side projected | Same init, input only | Same init, output only | Other init, input only | Other init, output only | Independent, input only | Independent, output only |
|---|---|---|---|---|---|---|
| Paraphrase | 0.995 | 0.72 | 0.65 | 0.715 | 0.65 | 0.71 |
| JSON extraction | 0.75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Arithmetic | 0.89 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Clinical extraction | 0.855 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Three points follow. First, the shared input subspace is the initialisation's: a basis from the
other initialisation captures almost nothing, even when it contains the same task trained from a
different start. Second, even within one initialisation, a subspace fitted on other tasks
captures only 3-12% of a new task's update, because the output side of a new task is not spanned
by the others (the side-split capture table above). Third, projection reduces the held-out tasks to base-model performance (zero-shot base: 0.65,
0.00, 0.00, 0.00). The other-initialisation and independent projections reproduce the base model's
prediction on every paraphrase example and score zero on the three generation tasks. The
same-initialisation projection keeps a small effect on paraphrase: 0.71 against 0.65 is a paired
difference of 0.06 (21 examples gained, 9 lost; exact McNemar p = 0.04; paired bootstrap 95% CI
0.01-0.115), far below the direct adapter's 0.99-1.00, and it too scores zero on the generation
tasks.

The one-sided projections locate the loss. With a basis from the same initialisation, projecting
only the input side keeps 95% of the update's energy and most of each task (0.75-0.995, against
0.90-1.00 for the direct adapter). Projecting only the output side keeps 7% of the energy and
returns every task to the base model's level, apart from a small paraphrase gain. Within one
initialisation, then, the loss comes from the output side. With a basis from another or an
independent initialisation, each side fails on its own. Input-only projection reproduces the base
model's prediction on every paraphrase example and scores zero on the generation tasks, and
output-only projection behaves as it does with the same initialisation. So a new task's input side
is spanned by other tasks only when they share its initialisation, and its output side is not
spanned in any condition. The small paraphrase gain survives every output-only projection (0.71-0.72
against 0.65; paired differences 0.06-0.07, exact McNemar p = 0.008-0.019, uncorrected for the three
comparisons), consistent with weak output-side structure shared across tasks. This experiment uses a small model and synthetic tasks, so it shows the mechanism rather
than replicating the paper; it does not test the paper's own held-out procedure, which may
re-fit coefficients rather than project.

### 3.4 Factorisation-invariant updates: input side and output side differ

| Top-16 share (uncentred, 15 slots) | Group | Singletons | Strength-matched null | Singletons, unit-norm | Null, unit-norm |
|---|---|---|---|---|---|
| Input side, layer 0 | 0.60-0.67 | 0.22-0.37 | 0.12-0.32 | 0.10-0.14 | 0.05-0.07 |
| Input side, layers 8-31 | 0.53-0.60 | 0.11-0.24 | 0.10-0.22 | 0.05-0.11 | 0.05-0.07 |
| Output side, layer 0 | 0.21-0.60 | 0.19-0.62 | 0.14-0.32 | 0.14-0.59 | 0.07-0.09 |
| Output side, layers 8-31 | 0.17-0.33 | 0.15-0.30 | 0.10-0.23 | 0.12-0.26 | 0.05-0.09 |

*Stability.* Over 20 subsamples of 193 of the 242 singletons (without replacement, fresh null
each time), the singleton excess over the strength-matched null is stable. Input side, layers
8-31: 0.002-0.020 (sd ≤ 0.001); output side, layers 8-31: 0.019-0.097 raw (sd ≤ 0.005) and
0.049-0.183 unit-norm (sd ≤ 0.006); layer 0 output q and k: 0.21 and 0.34 raw (sd 0.011).

*Input side.* The group remains far more concentrated than the singletons in the invariant
analysis (0.53-0.67 against 0.11-0.37), so the A-side result is not an artefact of analysing the
stored factors. Outside layer 0, singletons exceed the strength-matched null by only 0.003-0.022
(unit-norm: up to 0.04). Layer 0 shows more (0.034-0.103).

*Output side.* Group and singletons are similar (0.17-0.60 against 0.15-0.62), so the output-side
concentration is present in both inferred groups and does not track the shared initialisation. Singletons exceed the strength-matched null in every slot: by
0.019-0.106 outside layer 0 and by 0.22 (q) and 0.36 (k) in layer 0 (v: 0.047). With unit-norm
updates the gap widens (0.12-0.26 against 0.05-0.09 outside layer 0), so it is not produced by a few
large adapters.

*What the null contains.* Each ΔW is effectively low-rank (median participation ratio 2.1-5.6
across slots; the top singular value carries a median 34-67% of each update's energy), and adapter
norms vary widely (largest/median 2.7-7.8; 90th/10th percentile 5.9-9.3). These alone give
top-16 shares of 0.10-0.32, so equal-weight random baselines overstate how surprising the observed
concentration is.

### 3.5 How much of the remaining structure is the base model?

We compare the singletons' shared subspaces with the base weight's singular subspaces, and recompute
the excess over the null after projecting out the base weight's top-128 directions from both the
data and the null.

| Singletons | Overlap with base top-16 (× chance) | Energy in base top-128 directions: data (null) |
|---|---|---|
| Input, layer 0 q / k / v | 44× / 32× / 22× | 0.19 / 0.14 / 0.09 (0.03) |
| Input, layers 8-31 | 0.7-6.2× | 0.03-0.06 (0.03) |
| Output, layer 0 q / k / v | 17.6× / 12.4× / 1.8× | 0.43 / 0.71 / 0.17 (0.03 / 0.13 / 0.13) |
| Output, layers 8-31 | 0.8-3.0× | q 0.05 (0.03); k, v 0.12-0.14 (0.12-0.13) |

(Overlap is the mean cos² of principal angles between the two 16-dimensional subspaces; chance is
16/d. Energy is the share of the stacked rows' squared norm inside the base directions; the null's
share equals the chance level k/d.)

Projecting directions out changes the denominator of the top-16 share, so we report the excess
under two conventions. With a *fixed denominator*, the top-16 energy left after projection is
divided by the total energy before it. With a *renormalised* share, it is divided by the energy
left after projection.

| Singletons, excess over null | Before | After removing base top-128, fixed denominator | After, renormalised |
|---|---|---|---|
| Input, layer 0 q / k / v | 0.053 / 0.034 / 0.103 | −0.049 / −0.025 / 0.078 | 0.001 / 0.002 / 0.094 |
| Input, layers 8-31 | 0.003-0.022 | −0.004-0.020 | 0.000-0.021 |
| Output, layer 0 q / k / v | 0.220 / 0.361 / 0.046 | −0.028 / −0.111 / 0.033 | 0.177 / 0.146 / 0.048 |
| Output, layers 8-31 | 0.019-0.105 | 0.015-0.100 | 0.018-0.119 |

(10 null repeats; the null is the strength-matched null of Section 2, projected the same way.)

In layer 0 q and k the singletons put far more energy in the base weight's top directions than the
null does (on the output side of k, 71% against 12.5%). Removing those directions therefore takes
more energy from the data than from the null, and with a fixed denominator the excess turns
negative. With the renormalised share, the input excess in layer 0 q and k disappears, and the
output excess falls from 0.36 to 0.15 (k) and from 0.22 to 0.18 (q). The two conventions answer
different questions: how much of the original concentration survives, and how concentrated the
remainder is. Neither gives a percentage of the excess "explained" by the base weight, so we report
both rather than a single figure. In layer 0 the shared structure is strongly aligned with the base
weight's dominant directions on both sides.

Outside layer 0 the conventions agree. The singletons put roughly as much energy in the base
directions as the null (1.0-1.5 times on the output side, 1.0-1.8 times on the input side), and the
output excess changes little under either convention. The output-side shared structure outside
layer 0 is therefore present in both inferred groups, above a strength-matched null, and little
reduced by removing base-weight directions. We do not know what it is; candidates include
directions favoured by the training data format common to all Natural Instructions tasks, and
activation statistics that weight singular vectors do not capture.

**Learned movement of A.** Within the group, A minus the group mean is modestly more concentrated
than a strength-matched null (top-16 share 0.09-0.30 against 0.07-0.17; largest in layer 0), so
training moves A in partly shared directions, but this is small next to the initialisation.

### 3.6 The grouping also affects compression

Brüel-Gabrielsson et al. (2024), who trained this collection, serve many LoRAs by approximating
each update as U Σᵢ Vᵀ with shared bases U, V; they report that clustering adapters helps for
n ≥ 100 and use a reconstruction-loss threshold of 0.6 **[TODO: verify quotes and their error
definition]**. We re-implemented JD-Full (updates normalised to unit Frobenius norm, 10 alternating
iterations, no clustering) on 100-adapter sets, for 6 slots (layers 0, 16, 31 × q, v). We report
the mean relative squared Frobenius error 1 − ‖UᵀΔW_iV‖²_F per adapter; with unit-norm updates this
equals the squared error ratio of their Theorem 1.

| Rank | Group set | Singleton set | Mixed 50/50: group half | Mixed: singleton half |
|---|---|---|---|---|
| 16 | 0.64-0.86 | 0.85-0.90 | 0.67-0.83 | 0.87-0.98 |
| 32 | 0.44-0.75 | 0.72-0.81 | 0.51-0.77 | 0.70-0.80 |
| 64 | 0.25-0.59 | 0.50-0.65 | 0.34-0.64 | 0.41-0.60 |

Results are insensitive to the optimiser's initialisation (input-based vs random: ≤ 0.012), agree
across three restarts, match exact SVD to within 0.001 (checked on layer 16 q_proj), and converge
within about three iterations. At ranks 16-32 a shared basis fitted to a mixed set serves mostly
the group's adapters; by rank 64 the difference shrinks and reverses in some slots. Clusters found
in such a collection may partly reflect initialisation rather than task. We did not test serving
quality.

## 4. Discussion

**What the evidence supports.** On the adapters the paper lists, the A-side "universal subspace" is
produced by one large group of adapters with a common initialisation-like A, which training
changes only modestly. The effect persists in a factorisation-invariant analysis of the input
side. Without the group, the input side is close to a strength-matched null except in layer 0,
where the excess is aligned with the base weight's dominant directions. The output side tells a different story: a shared
structure present in both inferred groups, above a strength-matched null in
every slot and, outside layer 0, little reduced by removing base-weight directions. It is spread over far
more than 16 directions (k90 of B 140-1,200 on the full set).

**The strongest rebuttals.**
- *"Our exact analysis differs."* We used the paper's adapter list (the inferred fitting set; all
  502 give the same result) and its stacking convention;
  the full-set top-16 share is 0.44-0.46, and the knee at r = 16 is reproduced by a no-training
  simulation. "Most information in 16 or fewer directions" is therefore best read as describing
  this knee.
- *"One 16-dimensional input subspace serving 255 tasks is itself universality."* The subspace is
  a random one, and the other 242 adapters each use a different random one; any random
  16-dimensional input subspace appears to suffice. This is redundancy, consistent with frozen-A
  LoRA variants (Zhang et al., 2023, LoRA-FA) **[TODO: cite]**, not evidence of a privileged
  subspace. Whether task performance differs between the group and the singletons cannot be
  checked from published numbers (the model cards report none) **[TODO: or evaluate a sample]**.
- *"The output side is shared."* We agree, and treat it as the part of the claim that survives. It
  should be analysed separately from A and against strength-matched nulls.
- *"Low effective rank and uneven norms are themselves low-dimensional structure."* They are
  properties of individual adapters, not shared directions; Section 3.4 reports both baselines so
  readers can see what each removes.

**Recommendations for analyses of adapter collections.**
1. Report how initialisations were seeded, and check directly for shared initialisations
   (pairwise cosine of A suffices).
2. Analyse inferred initialisation groups separately, or remove the initialisation before pooling.
3. Use factorisation-invariant statistics of ΔW (uncentred C_in, C_out), and analyse input and
   output sides separately.
4. Compare against size-matched and strength-matched nulls, not equal-weight random matrices.
5. Apply the same checks to methods that exploit shared structure (joint-basis compression,
   clustering, continual-learning methods based on LoRA subspace angles).

**Limitations.** One collection and base model; 15 of 96 slots; the fitting/evaluation split of
Table 10 inferred from its colouring (including all 502 changes nothing material); the shared seed not recovered; base-model alignment measured
with weight singular vectors rather than activation statistics; compression re-implemented without
clustering or serving evaluation. Our claims concern the LoRA evidence only.

**A constructive direction.** Adapter geometry is only interpretable when the update shape is set
by the task rather than the seed. Task-gradient initialisation (LoRA-GA, Wang et al., 2024;
LoRA-One, Zhang et al., 2025) and preconditioned updates (Zhang & Pilanci, 2024) are candidates for
seed-invariant adapters; whether they deliver it is an open, testable question.

## 5. Related work

**[TODO: short paragraph.]** LoRA initialisation (PiSSA, LoRA-GA, LoRA-One); asymmetry between A
and B (Zhu et al., 2024) and frozen-A variants (LoRA-FA); recycling adapters into shared bases
(EigenLoRAx, 2025; Compress then Serve, 2024); shared LoRA subspaces in continual learning (O-LoRA;
arXiv 2602.06043).

## 6. Reproducibility

Code: `scripts/universal_subspace/` (`common.py` helpers; `analyse.py` grouping, spectra, nulls,
factorisation-invariant and base analyses, compression; `supplement.py` residual movement,
strength profiles and Figure 1 curves; `excess_subsample.py`; `grouping_all.py`; `figure1.py`;
`sensitivity.py` all-502 check, movement from the group mean and fixed-denominator base
decomposition; seed search scripts). Crossed experiment: `configs/experiments/dev_3080_crossed_*.yaml`,
`scripts/crossed_test.sh`, `scripts/crossed_eval.py`, `scripts/crossed_eval_sides.py` (one-sided
projections and paired tests). Outputs: `results/universal_subspace/` (`paper_set.json`,
`supplement.json`, `grouping_all.json`, `excess_subsample.json`, `sensitivity.json`) and
`results/crossed_eval/` (`crossed_eval.json`, `crossed_eval_sides.json`). The adapter list is
`results/universal_subspace/manifest_paper_v3.json`; all random generators are seeded and the
seeds recorded in the outputs. **[TODO: public repository link; make data paths configurable.]**

## References

**[TODO: complete and verify all entries.]**

- Brüel-Gabrielsson, R., Zhu, J., Bhardwaj, O., Choshen, L., Greenewald, K., Yurochkin, M., Solomon, J. (2024). Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead. arXiv 2407.00066.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv 2106.09685.
- Kaushik, et al. (2025). The Universal Weight Subspace Hypothesis. arXiv 2512.05117 (v3, 2026).
- Wang, S., et al. (2024). LoRA-GA: Low-Rank Adaptation with Gradient Approximation. arXiv 2407.05000.
- Wang, Y., et al. (2022). Super-NaturalInstructions. EMNLP 2022.
- Zhang, F., Pilanci, M. (2024). Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models. arXiv 2402.02347.
- Zhang, Y., et al. (2025). LoRA-One. arXiv 2502.01235.

## Revision log (v1 → v2)

Responses to the ChatGPT review (`review_gpt_note_lora_universal_subspace.md`) and the Claude review
(`review_note_lora_universal_subspace.md`):

| Point | Change |
|---|---|
| Paper's adapter list exists (v3 Table 10); headline not reproduced on our sample | All analyses rerun on the paper's set (497 after excluding 5 OOD); top-16 share 0.44-0.46; the knee is reproduced by a no-training simulation; the paper's set is shown to be enriched for the group (51% vs 38% collection-wide) |
| Stacking convention | Row stacking, as in the paper's Appendix B.5 **[TODO: confirm in v3]** |
| v3 acknowledges factorisation dependence | Quoted in the introduction |
| Centring breaks factorisation invariance | Invariant analyses now use uncentred C_in, C_out |
| Base-alignment numbers wrong (10-50×); causal claim too strong | Corrected (Section 3.5); excess after removing base directions now measured; conclusion limited to layer-0 q/k |
| Null at wrong N; "indistinguishable" | Size-matched null (N = 242, 10 repeats); wording "close to random, 1.3-3.3×" |
| Clusters are inferred, not seed histories; "untouched init"; distance from group mean | Terminology changed; kurtosis leads; distance described as from the group mean, a lower bound |
| "Almost all", "reproduces the spectrum"; 5% noise in the simulation | Simulation now pure, plus a movement-matched variant; output-side structure reported as real |
| Repeated nulls, bootstraps, unit-norm | 10 null repeats; unit-norm variants; subsample spread (20 × 80%, without replacement) |
| JD: seeding, restarts, exact SVD, init sensitivity, metric name | All added (Section 3.6); metric named precisely |
| Threshold sensitivity, all-slot agreement, saved diagnostics | Thresholds 0.15-0.6 and all 15 slots; all diagnostics saved to JSON |
| Provenance: manifest, no `exec` of external files | `common.py`, manifest input, seeds recorded |
| Seed in metadata? | Checked: none in model cards or configs |
| Learned movement of A in the group | Strength-matched null on the residual (Section 3.5) |
| What the null is made of | Effective ranks and norm spread reported (Section 3.4) |
| Crossed task × initialisation experiment | Run on Qwen2.5-0.5B (Section 3.3) |
| 903 vs 904; rank cap; d_out for k/v; define slot, top-16 share, k90; "gauge-free" | Fixed / defined in Section 2 |
| Performance by group | Not possible from published cards; listed as a limitation |

## Revision log (v2 → v3)

Responses to the second ChatGPT review (of draft v2):

| Point | Change |
|---|---|
| "Exact adapter set" is provisional | Called the *inferred fitting set* throughout; all 502 adapters checked: identical partition, no out-of-distribution adapter in the group, top-16 share −0.004 to −0.005 (Section 3.1) |
| Distance from the mean is not a per-adapter lower bound | Removed; only the aggregate statement kept (root-mean-square distance from the mean, 0.33-0.39, cannot exceed that from the initialisation) |
| Two-sided projection mixes input and output failure | Input-only and output-only projections added, with captured energy and task scores (Section 3.3): within one initialisation the input side alone keeps the task, the output side alone loses it |
| "Within sampling error" needs a paired test | Exact McNemar and paired bootstrap on the same examples for every paraphrase variant; the same-initialisation gain is small but significant (0.06, p = 0.04), and the text no longer claims equivalence |
| Base removal is not a percentage explained | Fixed-denominator and renormalised excess both reported, with energy in base directions (Section 3.5); no "explained" percentages |
| Title overstates | Retitled "Shared Initialisation Confounds the LoRA Evidence for Universal Weight Subspaces" |
| "Independent of initialisation" | Replaced by "present in both inferred groups" where the evidence is observational |

