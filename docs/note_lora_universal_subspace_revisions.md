# Revision logs: Shared Initialisation Confounds the LoRA Evidence for Universal Weight Subspaces

Responses to external reviews of [`note_lora_universal_subspace.md`](note_lora_universal_subspace.md),
moved out of the manuscript at v3.2.

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

## Revision log (v3 → v3.1)

Responses to the third ChatGPT review (of draft v3):

| Point | Change |
|---|---|
| "Not spanned in any condition" is too broad | All projection conclusions restricted to the fitted rank-16 bases (Section 3.3) |
| Bases fitted from raw A/B factors are factorisation-dependent | Added bases from the invariant operators C_in and C_out; captured shares and task scores agree with the paper-style bases to within 0.03 |
| Arithmetic 0.01 contradicts "zero on all generation tasks"; opening with "reduces to base" | Generation scores stated as 0-0.01; the paraphrase gain is described before the comparison with the base model |
| Only initialisation 0 held out | Initialisation 1 now held out as well, with both basis types; results match. Seed coverage stated in Section 3.3 and the limitations |
| "Any random subspace appears to suffice"; "B's spectrum does not depend on group membership" | Now "the tested initialisations" and "similar spectra in both groups" |
| (Own check) | Two-sided projections rescored on the 3080: same score on 197-200 of 200 examples per task |

## Revision log (v3.1 → v3.2)

Responses to the fourth ChatGPT review (of draft v3.1):

| Point | Change |
|---|---|
| "Paper-style" projection bases were uncentred | Three bases now: centred factor (the paper's PCA convention), uncentred factor, and invariant. Centred factor bases were scored on four tasks and give the same pattern; median captured shares within 0.037 of the invariant bases |
| Evaluation cache ignored rank | Every evaluation is named by a fingerprint of its full configuration (rank, basis, side, adapter SHA-256s, evaluation settings, projection code version) and stored with that configuration; a score is reused only for an identical configuration. All 250 scores were recomputed from scratch in one run, base model and direct adapters included, and a full rerun reused every one |
| Singletons "each start from a different" initialisation | Now "show no detected shared initialisation under our grouping criterion" |
| Knee versus majority | Now "We reproduce a rank-16 knee, but not a majority of the variance within 16 components" |
| Why four of ten tasks were scored | All ten tasks are now scored with invariant bases. The four-task subset (used for centred factor bases) was fixed in the script before any projection was computed; this is documented in Appendix A. The ten-task results add one uninformative task (hidden-rule sentiment was not learned) and two tasks whose output-side projection keeps part of the effect (paraphrase, concise rewriting) |
| Task-score table hard to read | Tasks in rows; full condition matrix moved to Appendix B |
| References, methods, release, TODOs, review logs | References verified and completed; quotes checked against the papers (one is a figure caption); stacking convention confirmed from Appendix B.5; Compress-then-Serve threshold described as stated there; methods appendix added; model names corrected to the Instruct variants; data paths configurable through environment variables; repository linked; review logs moved to this file. Remaining: check Appendices B.7 and C.6 of the paper for seeds or baselines, and tag the posted commit |

## Revision log (v3.2 → v3.3)

Responses to the fifth ChatGPT review (of draft v3.2):

| Point | Change |
|---|---|
| Abstract should mention the output-side exception | Added: the fitted output bases retain little update energy but preserve partial performance on paraphrase and concise rewriting |
| Qwen and Mistral residual structures are not the same phenomenon | Section 3.3 and the discussion now say the Qwen results show that shared output directions can carry task information in that model, without establishing what the Mistral residual represents |
| Same seed guarantees identical A only with the same generator state; "training changes A only modestly" is inferred | Both qualified: identical A requires the same generator state and draw sequence; modest change is inferred from closeness to the group mean, not measured against the unavailable initialisation |
| Cache guarantee too broad | The note and script now list exactly what the fingerprint covers. A new provenance record covers what it omits (adapter configs and LoRA scale, test-split hashes against the committed manifests, resolved base-model revision) and confirms none changed after the run began |
| Remaining TODOs | Appendices B.7 and C.6 checked: no seeds or random baselines. B.7 turned out to describe the paper's own held-out reconstruction (projection with "analytically reconstructed" coefficients, up to 450 fitting adapters, Rouge-L 72.3 against 73.7). All nine held-out tasks are singletons under our grouping; Section 3.3 now compares the two settings. Code and results cited by immutable commit. The author line is left for the author |
