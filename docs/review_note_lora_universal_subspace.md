# Review of `note_lora_universal_subspace.md`

*2026-10-08. Skeptical ML-reviewer pass over the draft note, the detailed check
(`universal_subspace_check.md`), the code in `scripts/universal_subspace/` and the JSON outputs in
`results/universal_subspace/`. The raw adapters live on the Windows `F:` drive, which is not mounted
in WSL, so I could not re-run the seed grouping; I checked every number in the note against the JSON
and re-derived the random-matrix baseline independently.*

## Verdict in one paragraph

The central result holds and is important: 197 of 500 public adapters share one random A, that group
alone produces the low-dimensional A spectrum, and the matched control (3.2) is clean. The code for
seed grouping, the factorisation-invariant SVD and the strength-matched null is correct. The problems
are in framing and in a few numbers. The note never says plainly that it does **not** reproduce the
paper's headline on its own sample (34% in the top 16, not "most"), it has not pinned down whether the
paper stacks the same way, Section 3.5 generalises a layer-0 result to the whole model and quotes a
wrong ratio, and "indistinguishable from random" is compared against a baseline at the wrong sample
size. None of these overturn the conclusion; all of them would be caught by a referee.

## Must fix

1. **State that the paper's headline is not reproduced, and pin down the analysis convention.**
   On the full 500 the note gets 33-35% in the top 16 components and about 1,900-2,000 components
   for 90% of the variance (`subspace_N500.json`, `raw_all`). The paper says "most variance in 16 or
   fewer directions". That gap is currently a TODO; it has to be the first thing addressed, because
   two explanations change the argument:
   - *Different stacking.* The note stacks rows (N·16 × 4096). If the paper instead treats each
     adapter's A as one flattened sample (500 × 65,536) and runs PCA across adapters, 197 identical
     copies produce **one** dominant direction, not 16, and the "drop at exactly the LoRA rank"
     argument in 3.3 does not apply. The seed story still explains the result, but through a different
     signature. Check the paper's appendix/code and run both conventions.
   - *Different adapters.* In the sample, the shared-seed fraction falls with task number: 58% of
     sampled tasks below task 200 share the seed, 44% in 200-400, and 25-34% above task 1000. If the
     paper's 500 were low-numbered tasks the shared fraction could be near 50%, giving a top-16 share
     near 0.5. Still not "most". Say this, and extend the grouping to all 903 adapters (only the
     layer-0 q A is needed per adapter).

2. **Section 3.5 overclaims "tied to the base model", and the "10-50×" is wrong.**
   From `basecheck.json`, output-side overlap with the base weight's top-16 directions:

   | Slot | Overlap | Chance | Ratio | Energy in base top-128 | Chance | Ratio |
   |---|---|---|---|---|---|---|
   | L0 q | 0.063 | 0.0039 | 16× | 0.39 | 0.03 | 12× |
   | L0 k | 0.193 | 0.0156 | 12× | 0.67 | 0.125 | 5.4× |
   | L0 v | 0.031 | 0.0156 | 2× | 0.17 | 0.125 | 1.4× |
   | L8-L31, all | 0.009-0.032 | | 0.8-3.3× | | | 0.9-1.6× |

   The range "0.06-0.19 in layer 0" silently drops L0 v (0.031). "10-50×" matches no slot; 50 is
   what you get dividing k's overlap by q's chance. Correct is about 12-16× for q and k and 2× for v.
   More importantly, the output-side excess over the strength-matched null is **present in every one
   of the 15 slots** (absolute 0.02-0.10 outside layer 0, i.e. 1.1-1.8× the null), and in layers
   8-31 the base weight explains none of it (overlap 1-3× chance, energy at chance). So the honest
   statement is: the layer-0 q/k excess follows the base model; the smaller excess everywhere else
   is real and unexplained. Reword the 3.5 heading, the abstract ("large only in the first layer,
   where it aligns with the base model") and the Discussion accordingly.

3. **"Indistinguishable from independent random matrices" uses the wrong baseline.**
   The note compares own-seed adapters (N = 197, top-16 share 0.022-0.055) against 0.011, which is
   the iid baseline at N = 500. At N = 197 the iid baseline is 0.017 (k90 ≈ 1,809; I recomputed
   this). Own-seed adapters are therefore 1.3-3.2× random, highest in layer 0 (q 0.055, k 0.035,
   v 0.047). "Indistinguishable" should become "close to random" with the matched-N number, or
   better, the strength-matched null applied to the paper-style A spectrum. Same fix in the abstract
   ("the level of independent random matrices"). Also 0.055 rounds to 0.06, not 0.05.

## Should fix

4. **The learned part of A in the shared group is itself concentrated, and the note skips it.**
   Subtracting the group mean lowers the top-16 share to 0.10-0.31, but k90 is 280-750 against
   about 1,800 for random at this N. A referee will read that as "once you remove the init, the
   *trained* movement of A lives in a few hundred shared directions, 3-6× more concentrated than
   random". Either apply the strength-matched null to the residual (per-adapter movement is rank
   ≤ 16 and norms vary) or discuss it explicitly.

5. **The no-training simulation is not "pure Kaiming initialisations".** `subspace.py` line 102 adds
   5% iid noise to each adapter (`+ 0.05 * kaiming(...)`). Say so, and explain why the real
   collection is lower (0.34 vs 0.39): training moved A by a median 23%, spreading variance.

6. **Compression metric and normalisation need to match Compress then Serve before quoting 0.6.**
   `robust.py` reports 1 − ‖UᵀΔW V‖² on unit-norm updates, which is the *squared* relative Frobenius
   error. If the paper's 0.6 threshold is unsquared, the note's 0.86-0.90 corresponds to 0.93-0.95
   and 0.62 to 0.79. State the definition in the note and verify the paper's. Also confirm the
   paper normalises each update to unit norm; if not, say the normalisation is yours.

7. **JD-Full uses randomized SVD, which biases the own-seed sets the wrong way.** `top` uses
   `torch.svd_lowrank(q=R+16, niter=4)`. For own-seed sets the spectrum is flat and randomized SVD
   is least accurate there, inflating own-seed errors, i.e. in the direction of the conclusion. The
   matrices are only 4096 × 1,600; use exact `torch.linalg.svd` and confirm the numbers are
   unchanged. Not a bug in logic, a possible bias in the numbers.

8. **Report what the null is made of.** The null reaches 0.10-0.34 top-16 share from strengths
   alone, which means each adapter is effectively low rank (a few dominant singular values) and
   adapter norms vary widely. Give the median per-adapter effective rank (or share of the top
   singular value) and the spread of adapter norms. Without this the reader cannot judge the null,
   and the paper's authors could reasonably call "effective rank ≈ 4" a low-dimensional finding in
   itself.

9. **Several reported numbers exist only in stdout, not in `results/`.** The pairwise cosines
   (min 0.62, median 0.90, between-group 0.003 / max 0.11), movement percentiles (23%, 7-62%),
   std/kurtosis, grouping agreement and all seed-search results are printed, not saved. Save logs.
   Also "identical in every layer and module we checked" was three slots (L0 q, L16 k, L31 v) via a
   pairwise-agreement average dominated by non-pairs; say "three slots" and give the number.

10. **The group mean is a biased estimate of the init.** The mean of 197 trained A matrices absorbs
    any training direction common to the group, so the 23% movement is a lower bound and the std
    match (0.00902 vs 0.00902) is uninformative: averaging 197 deltas inflates the std by under
    0.1%. The kurtosis test is the real evidence. `distcheck.py` also computes the sharpest test,
    max|x| ≤ 1/64 and the fraction of entries outside the uniform bound; report it.

11. **Two different random own-seed subsets are used.** `subspace.py` draws with seed 1,
    `robust.py`/`basecheck.py` with seed 2. Results are consistent, but say it or unify.

12. **"Maximum 3,152 components" is wrong for the B matrices of k and v.** Mistral's k and v
    projections have d_out = 1,024 (grouped-query attention), so at most 1,024 components. State
    d_out for k/v, since chance levels and k90 ceilings differ from q.

13. **Cheapest possible check not done: look for a seed in the metadata.** The Lots-of-LoRAs model
    cards list training hyperparameters; `adapter_config.json` and the README may contain a seed
    field. One grep over the 500 configs could settle Section 3.1 and replace the seed search.

## Minor

- Abstract: "still 90% identical (median cosine)" is loose; write "median pairwise cosine 0.90".
- 3.4: "within 0.04 in all slots but one" — L0 q is 0.042. Say "≤ 0.03 outside layer 0 (L0 q 0.04,
  L0 v 0.08)".
- 3.4 table mixes a 6-slot row and 15-slot rows, with blanks in the shared-seed column. Explain that
  the shared-seed factorisation-invariant analysis was run for 6 slots (layers 0, 16, 31; q, v) and
  the null for 15, or split into two tables.
- 3.6: mixed-set own-seed error of exactly 1.00 (L16 q, rank 16) looks like a bug to a reader; add a
  sentence that it means the shared basis captures essentially nothing of the own-seed half.
- `basecheck.py` imports `robust.py` with `exec(open(...).read().split(...))`; hard-coded Windows
  paths throughout. Fine for now, but the reproducibility section promises more.
- Remaining TODOs: paper quote, CtS quotes, adapter list, Figure 1, references (Kaushik et al.
  author list, LoRA-One authors). A figure of the cumulative-variance curves (real, simulation,
  own-seed, iid) would carry most of Section 3.3 on its own.

## The authors' strongest rebuttal, and whether the note answers it

**(a) "You analysed different adapters with a different stacking and got 34%, so your critique is
of your own analysis, not ours."** Not yet answered. This is why item 1 is must-fix. Until the
convention and adapter set are pinned down, the paper can say the note did not reproduce it.

**(b) "The shared-seed adapters really do share a 16-dimensional input subspace in ΔW, and they
work on 197 different tasks. One fixed subspace sufficing for 197 tasks is exactly the universality
we mean."** The note's own Section 3.4 confirms the shared subspace is in ΔW, not just in A, so
this rebuttal is available to them. The answer the note should make explicit: the subspace is a
*random* one, and the 303 own-seed adapters each use a different random one. That shows any
random 16-dimensional input subspace suffices (consistent with frozen-A LoRA variants and with this
project's own frozen-A results), which is a statement about redundancy, not about a privileged
shared subspace. The decisive check is whether task performance depends on the seed group; the
model cards report per-task evaluation numbers, so this is cheap. The Discussion currently gestures
at this but does not say it.

**(c) "Your null is too strong. Low effective rank per adapter and uneven adapter norms are
themselves low-dimensional structure."** Partly answered: the note is right that "shared subspace"
means shared directions, and the null isolates directions. But it should present both the
equal-weight and the strength-matched baselines side by side and report the effective ranks
(item 8), so the reader sees what each baseline removes.

**(d) "The output-side excess over your null is consistent in all 15 slots, and B's k90 is
110-800 in a 1,024- or 3,152-dimensional space. That is a shared subspace; we never said it had to be
16-dimensional."** The note's answer ("small, tied to the base model") only holds for layer-0 q and
k (item 2). For the other 13 slots the note has no explanation and should say so rather than fold
them into the base-model story.

## Missing experiment that would most strengthen the note

Run the paper's exact analysis (its code if released, otherwise both stacking conventions) on
(i) the note's 500, (ii) all 903 adapters, and (iii) the lowest-numbered 500, reporting the
shared-seed fraction and the top-16 share for each. This closes rebuttal (a) and turns the
TODO into a result. Second priority: the performance-by-seed-group check from (b). Third: a
strength-matched null for the paper-style A and B spectra and for the shared group's residual
(items 3, 4), so one baseline is used throughout.

## Clarity for a reader who knows LoRA but not this project

- Define "top-16 share" and "components for 90%" once, with the formula, and use the same two
  names everywhere (the note alternates "comps.", "k90", "directions").
- Say up front that A is r × d_in (input side) and B is d_out × r (output side), and that
  d_out = 1,024 for k and v. Readers will otherwise assume 4096 everywhere.
- "Gauge-free" is not standard LoRA vocabulary; "factorisation-invariant" or "computed from ΔW = BA
  rather than from A and B separately" is clearer. Keep one term.
- "Slot" should be defined as one (layer, module) pair, and the note should state which slots each
  analysis used: 15 for the paper-style spectrum and null, 6 for the shared-seed ΔW analysis and
  compression.
- The intro cites "4-22%" and "97-99%" from this project's own experiments without saying how
  movement was measured (relative Frobenius distance from the init) or what "overlap" means.
- The shared-seed group is described as "39%" in the abstract and "197 of 500" in 3.1; the
  simulation's 39% in 3.3 is a different quantity that happens to match. Point out that under row
  stacking the top-16 share of the no-training simulation is approximately the shared-seed
  fraction, which is why they agree. That is a useful intuition, not a coincidence.

## Numbers checked against the JSON

All ranges in Sections 3.2, 3.3, 3.4 and 3.6 match `subspace_N500.json`, `robust.json` and
`basecheck.json` to the stated precision, with these exceptions: 0.055 rounded to 0.05 (3.2),
"within 0.04" with L0 q at 0.042 (3.4), the L0 v overlap omitted from "0.06-0.19" and the "10-50×"
ratio (3.5). Numbers in 3.1 (cosines, movement, std, kurtosis, seed search) are not in any saved
output and could not be checked.
