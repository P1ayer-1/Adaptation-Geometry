# Proposed amendments to the RSI plan's evaluator ("Holdout")

*From the Adaptation Geometry project, 2026-10-08. Evidence: [`dev_scale_report.md`](dev_scale_report.md).
Each amendment states the rule, what went wrong without it, and how to apply it. Numbers come from
Qwen2.5-0.5B and Llama-3.2-1B; the failure modes are general, the exact thresholds are not.*

## 1. Gains must beat a few-shot baseline (skill-not-format gate)

**Rule.** Count a training gain as a new skill only if the trained model beats the *untrained model
given a few worked examples in the prompt*, not the untrained model with a bare prompt.

**Why.** On our first task panel, 5 worked examples already brought untrained models to 0.98-1.00
on JSON extraction and arithmetic. The apparent LoRA "lift" was almost all output format. After
redesigning the tasks so examples cannot reveal the rule, 5-shot scores fell to 0.00-0.60 while
trained adapters reached 0.90-1.00, which is a real skill gap.

**Apply.**
- Every Holdout task records a k-shot score (k = 5 worked) for the current student, and lift is
  measured from it.
- Flag cells where the k-shot score is already near ceiling: those tasks cannot show skill gain and
  should not count toward promotion.
- Recalibrate against the largest student in use. A task that is hard for a small student may be
  format-only for the next generation.

## 2. Kill rules must allow late take-off

**Rule.** Judge a plateau by the amount of data seen (passes over the task data), not by steps,
turns or wall-clock. Never stop or kill a run on a hard task before it has had a minimum exposure.

**Why.** Two of our ten tasks sat at chance for about 6,000 examples and then learned quickly to
0.92-1.00. Early stopping with patience counted in steps killed them just before take-off, which
also made the outcome look seed-dependent. With a frozen adapter basis, take-off came at 0.8-1.1
passes.

**Apply.**
- Minimum exposure before any kill: at least two full passes over the task data (or the
  equivalent number of examples for generated tasks).
- Patience measured in passes, not steps.
- Keep at least one known late-take-off **canary task** in every run. If the canary is killed before
  it takes off, the kill rules are too aggressive. Treat that as an evaluator bug, not as a result.

## 3. Transfer claims must beat the target's own average adapter

**Rule.** Any claim that knowledge carried over (from another model, another generation, or a
library of adapters) must beat a baseline that uses no information from the source: the target
model's own average update over its other tasks.

**Why.** That average carried 13-28% of the benefit to unseen tasks, while every learned
cross-model map recovered only 1-6%. Without this baseline, a map that only learned "this is what
fine-tuning usually does to this model" would look like successful transfer.

**Apply.** Report three numbers for every transfer claim: no update, target mean update, transferred
update. A transfer method passes only if it beats the target mean.

## 4. Graded likelihood next to pass/fail

**Rule.** Alongside each pass/fail task metric, record how much more likely the gold answer became
(share of the gold-answer likelihood gain that direct training achieves; we called this
RecoveredNLL).

**Why.** Exact-match scores showed 0.000 for every transferred update, which hides the difference
between "slightly helpful", "useless" and "actively harmful". The graded measure showed transferred
updates were often harmful (RecoveredNLL down to −4.8), as bad as a random update of the same size.

**Apply.** Holdout stores both scores. Promotion uses pass/fail; diagnosis and transfer decisions use
the graded score. A negative graded score is a regression even if pass/fail did not move.

## 5. Carry skills across student generations as data, not adapters

**Rule.** When moving from one student to the next (stage 3 → 4 → 5), carry skills forward as
training data, environments and verifiers. Do not port adapters or adapter codes between models.

**Why.** Three independent methods (weight-space maps, maps with a frozen basis, a shared adapter
with per-model connectors) recovered only 0.4-6% of the benefit of training the new model directly.
As a head start with 32-512 target examples, a transferred adapter helped in 1 of 12 settings and
hurt in 7. Plain training from scratch on the same few examples beat it on the hard task by a wide
margin (0.67-0.93 against at most 0.53).

**Apply.** Every skill the system acquires must leave behind a re-runnable artefact (dataset,
environment, verifier) so the next student can learn it from scratch. Adapters are per-model caches,
thrown away at a generation change. Revisit only if a head-start test on the real students shows a
reliable gain.

## 6. Control seeds before reading anything into adapter geometry

**Rule.** If any loop compares adapters by angle or overlap (for routing, deduplication,
consolidation or forgetting prediction), give each adapter its own random initialisation, or freeze
one basis per model by design, and check the comparison against a seed-only control.

**Why.** LoRA's input side (the A matrix) moves only 4-22% from its random starting point. With the same seed, adapters
for unrelated tasks showed 97-99% overlap that came entirely from the shared starting point. With
independent seeds, two runs of the *same* task had full-update cosine of only about 0.08.

**Apply.** Before using a geometric similarity in a decision, compute it for (a) two seeds of the
same task and (b) two unrelated tasks with shared seeds. If (b) is high or (a) is low, the measure
is reading the seed, not the task.
