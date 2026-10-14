# Review of `note_lora_universal_subspace.md` (ChatGPT)

*2026-10-08. External review pasted into the working session by the user; kept for the record.
Responses: see the revision log at the end of `note_lora_universal_subspace.md`.*

## Must fix

1. **The paper's adapter list is available (v3, Table 10); resolve the replication discrepancy.**
   The v3 paper (Oct 5, 2026) lists model IDs, including evaluation models to distinguish from the
   fitting collection, and explicitly acknowledges that its factor analysis is not gauge-invariant.
   The "regardless of initialization, task, or domain" quotation is accurate. The note's own sample
   gives a top-16 A share of only 33-35%, so it does not reproduce the headline before any control.
2. **Centring the SVD factors breaks the claimed gauge-free statistic.** The QR/SVD factorisation in
   `robust.py` is correct, but its spectrum centres the stacked s_j v_j / s_j u_j. SVD vectors have
   arbitrary paired signs, which change the row mean. Use the uncentred operators
   C_in = Σ ΔW_iᵀΔW_i, C_out = Σ ΔW_iΔW_iᵀ (as `basecheck.py` already does).
3. **Base-alignment numbers are wrong and the causal conclusion too strong.** Layer-0 output overlap:
   q 0.063 (16.1×), k 0.193 (12.4×), v 0.031 (2.0×), not "0.06-0.19, 10-50×". Alignment with base
   directions does not show the residual is "a property of the base model rather than a task-general
   adaptation subspace"; measure how much excess over the null disappears after removing the
   base-aligned component.
4. **"Indistinguishable from random" is unsupported.** Own-seed (N = 197) 0.022-0.055 vs a random
   baseline computed at N = 500 (0.011). Generate the null at N = 197, repeat, report uncertainty.
5. **Observed clusters are not recovered seed histories.** Use "inferred shared-initialisation group"
   and "singleton group"; matching moments supports an initialisation-like component, not an
   "untouched" init; distance from the trained group mean is not movement from the true init;
   subtracting the mean removes shared learned change too; "90% identical" → "median pairwise
   cosine 0.90".
6. **"Almost all" and "reproduces the spectrum" overstate the controls.** Real 0.330-0.349 vs
   simulated 0.389-0.392; the simulation adds 5% noise. Layer-0 output gaps of 0.196 (q) and
   0.345 (k) are substantial exceptions.

## Should fix

7. Repeated null rotations, adapter bootstraps, unit-norm results; a strength-preserving null tests
   orientation conditional on strength and cannot show training learned no shared structure.
8. JD: seed `svd_lowrank`; convergence traces, restarts, exact-SVD comparison; init sensitivity;
   label the metric as mean relative squared Frobenius error and check the 0.6 convention.
9. Grouping: report threshold sensitivity and full within/between distributions; agreement was
   checked at only two additional slots and only printed.
10. Provenance: save cosine distributions, moments, movement quantiles and seed-search results;
    use the manifest instead of a sorted glob; replace `exec` of external files.

## Minor

0.042 is not "within 0.04"; 904 unique IDs, not 903; centring caps rank at 3,151; define "slot"
and list slots per analysis; state that the null uses 197 of 303 singletons; explain one-sided
spectra; "B's concentration is not a seed effect" too categorical (differences up to 0.041); link
and define the 40-run experiments; note α/r = 2 cancels in normalised metrics.

## Strongest rebuttal and missing experiment

Rebuttal: no exact replication; factorisation dependence already acknowledged; the full-update
analysis retains shared output structure; alignment with a common pretrained model could be
meaningful cross-task regularity; reconstruction error does not show usefulness.

Experiment: a crossed task × initialisation design (same tasks under several shared and
independent A initialisations, everything else fixed, A0 saved); fit subspaces on some tasks,
evaluate held-out tasks and held-out initialisations by captured update energy and downstream
performance after projection.

Strongest defensible conclusion today: "Strong A-side concentration in this sample is associated
with an inferred shared initialisation and can arise without task learning; interpreting it as
learned universality requires additional controls."
