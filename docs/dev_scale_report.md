# Do task adaptations share a geometry across models? Dev-scale findings

*Universal Adaptation Geometry, October 2026. Small-model experiments before Stage 0.
Code, configs and raw results: this repository (branch `claude/task-design-fixes`).*

## Summary

The project asks whether LoRA task adaptations live in a shared, model-independent geometry:
whether there is a task representation *z_t* and per-model decoders *D_m* with
Δ(m,t) ≈ D_m(z_t), such that a **new** task learned on one model can be carried to another
model without training the target on that task.

On two small instruction-tuned models (Qwen2.5-0.5B and Llama-3.2-1B) and ten rule-generated
tasks, the answer for that strong form is **no**:

- **Within a model, adaptations are highly reproducible.** With one fixed LoRA A matrix per
  model, two seeds of the same task agree at a global cosine of 0.92-0.93, against 0.08 with
  standard LoRA.
- **Across models, a held-out task barely transfers.** Three independent approaches agree.
  Weight-space maps, frozen-A maps and an end-to-end shared adapter recover only about 0.4-6%
  of the benefit of training the target directly. Measured as gold-answer likelihood, that
  benefit is nowhere near enough to change task scores.
- **More training tasks help only slowly.** Each task occupies mostly its own directions.
  Coverage of a held-out task grows by about 2-4% per added training task. A learned map's
  transfer went from 2.3% to 3.2% as training tasks grew from 2 to 8.
- **The strongest non-learned baseline is not a random update.** It is the target model's
  own average adapter, which ignores the source model and recovers 13-28%.

Along the way, the work produced several methodological findings that matter for any study of
adapter geometry. They are listed in the [last section](#methodological-lessons).

## Setup

| | |
|---|---|
| Models | Qwen2.5-0.5B-Instruct and Llama-3.2-1B-Instruct, frozen, bf16 |
| Adapters | LoRA rank 16, alpha 32, on q, k, v, o, up, down, gate in every layer |
| Tasks | 10 rule-generated tasks (panel v2), 8,000 training examples each; gold answers score 1.0 by construction |
| Held-out tasks | T4 (nested JSON extraction with hidden conventions) and T8 (3-5 step arithmetic) |
| Baseline for "lift" | the model with 5 worked examples in the prompt (few-shot), not the bare prompt |
| Main transfer measures | task metric on 200 test examples; RecoveredNLL = share of the gold-answer likelihood gain of direct training that a transferred update recovers |
| Hardware | RTX 3080 laptop (8 GB), rented 4× RTX 5090 and 2× A100 |

## Findings, in the order they were found

### 1. The first task panel measured output format, not skill

On the v1 panel, 5 worked examples in the prompt already brought the untrained models to
1.00 on JSON extraction and 0.98-1.00 on arithmetic, and sentiment sat at 0.90-0.95
zero-shot. The LoRA "lift" was almost entirely output format. Lift is now measured against
the few-shot model, and ceiling and format-dominated cells are flagged. The v2 panel was
redesigned so that 5 demonstrations cannot reveal the rule. Its 5-shot scores are 0.00-0.60
on every task, and direct LoRA reaches 0.90-1.00.

### 2. Standard LoRA's input side stays at its random initialisation

LoRA trains ΔW = B·A from a random A and B = 0. Across 40 runs, A moved only 4-22% from its
initialisation, and 96-99.8% of its input directions were still the original random ones.
Consequences:

- **Shared-seed confound.** With the default seeding, every task on a model shares the same
  random A for a given seed. Unrelated tasks then share 97-99% of their input directions, an
  artefact that would make held-out tasks look predictable. Per-task initialisation removes it.
- **Task signal sits only on the output side.** Two seeds of the same task agreed on output
  directions (subspace overlap 0.27-0.30 against 0.04-0.08 for unrelated tasks), but their
  input directions overlapped only at chance.

### 3. Weight-space maps cannot transfer a held-out task's input side

Maps were fitted from 8 training tasks in truncated-SVD coordinates (Procrustes and bilinear).
They predicted updates with relative error exactly 1.000, i.e. almost no update at all, and the
predictions scored like the untrained model. A held-out adapter's input side lay inside the
training tasks' span only at chance level (0.13 vs 0.126 for Qwen, 0.06 vs 0.056 for Llama).

### 4. A frozen, shared A per model makes adapters reproducible, but transfer still fails

Freezing one random A per model (LoRA-FA style, shared by all tasks and seeds) gives:

| | Trainable A | Frozen A |
|---|---|---|
| Direct LoRA test scores | 0.90-1.00 | 0.90-1.00, within 0.03 everywhere |
| Same task, two seeds: global cosine | 0.08 | **0.92-0.93** |
| Held-out output side inside the training span (k=16) | 6-16% | 6-16% |
| Learned map, RecoveredNLL | 0-2% | 1-6% |
| Target's own mean adapter, RecoveredNLL | up to 28% | up to 21% |

Freezing A moves the map's representable ceiling from about 0.1-0.4% to 6-16%. The output side
then becomes the bottleneck: a new task writes mostly into directions the training tasks never
used. A second seed of the *same* task lies 94-96% inside the first seed's output directions.

### 5. More training tasks help only slowly

With maps fitted on nested sets of 2, 4, 6 and 8 training tasks:

| Training tasks | Held-out coverage, all directions (Qwen / Llama) | Learned map RecoveredNLL | Target mean RecoveredNLL |
|---|---|---|---|
| 2 | 0.16-0.20 / 0.07-0.10 | 2.3% | 13% |
| 8 | 0.42-0.46 / 0.19-0.24 | 3.2% | 16% |

Coverage grows roughly linearly, with no sign of saturating near full coverage. The maps'
16-direction coordinate system barely improves.

### 6. Hard tasks take off late; frozen A learns them too

Two tasks sit at chance for about 6,000 examples and then learn quickly: T1 (a hidden rule,
review polarity XOR a hidden split of 40 products) and T2 (transitive relational NLI). Early
stopping with short patience had stopped them before the take-off, which made learning
seed-dependent. Requiring at least one full pass rescued all four failed runs (0.92-1.00).

Frozen A takes off at 0.8-1.1 passes, as fast as trainable A, so it needs at least two passes
before stopping is allowed. Data-informed fixed A matrices did not consistently improve speed
or final scores, so plain frozen random A is enough:

| Fixed A | T1 test (Qwen / Llama) | T2 test | T3 test |
|---|---|---|---|
| Random | 1.00 / 1.00 | 0.93 / 0.95 | 0.94 / 0.98 |
| Whitened on generic text | 0.97 / 1.00 | 0.91 / 0.97 | 0.99 / 0.95 |
| PCA on generic text | 1.00 / 0.98 | 0.91 / 0.91 | 0.96 / 0.96 |

### 7. A shared adapter trained end to end is expressive but not portable

The design has, for every model, learned connectors into and out of a 64- (or 128-) number
shared code, and for every task a shared core matrix per layer slot. Phase 1 trains the
connectors and the training tasks' cores on both models at once. For a held-out task, a core is
then learned on one model with the connectors frozen, and plugged into the other model with no
training.

| Width 64 | T4 → Qwen | T4 → Llama | T8 → Qwen | T8 → Llama |
|---|---|---|---|---|
| Core trained on the target itself (ceiling), task score | 0.995 | 0.985 | 0.885 | 0.870 |
| Core transferred from the other model, task score | 0.000 | 0.000 | 0.000 | 0.000 |
| Gap closed by the transferred core, relative to the untrained core | 0.7% | 0.6% | 5.2% | 3.8% |

The connectors can express new tasks: a new 64×64 core alone reaches 0.79-1.00 on both models.
But a core's meaning does not carry across models. Width 128 behaves the same, closing
0.4-1.7% of the gap. With identity-initialised cores, the connectors also learn a default update
from the training tasks, and that default degrades held-out tasks well below the untouched model.
A zero-default variant removes this flaw; see "Pending" below.

### 8. Merging updates into bf16 weights does not distort evaluation

Applying a ΔW by merging it into bf16 weights gives the same task scores and the same
gold-answer likelihood (within a few percent) as merging into fp32 weights.

## Interpretation

**What failed.** The strong hypothesis fails at this scale: a new task's adaptation does not live
in a cross-model geometry that can be learned from other tasks. Adaptations look less like points
in a small shared space and more like stable, largely task-specific directions in each model.
The part that is shared is mostly a generic "this model was fine-tuned" component, which the
target model's own average adapter captures without any information from the source model.

**What is not ruled out:**

- **A weaker hypothesis.** A shared task code may port across models when the target gets a
  little data for that task. PorTAL (Ramp Labs, 2026) reports this for held-out models and known
  tasks. Our results concern held-out tasks with no target data, and are consistent with it.
- **Scale.** These results come from 0.5B and 1B models, a single model pair, and at most 8
  map-training tasks. Larger models, or many more training tasks, might share more structure.
  The linear growth of coverage suggests dozens of training tasks would be needed.

## Methodological lessons

1. **Measure lift against a few-shot baseline.** Otherwise an adapter that only teaches output
   format looks like skill.
2. **Do not share LoRA's random A across tasks.** Standard seeding does this for a given seed and
   creates spurious cross-task geometry. Either initialise per task, or freeze one A per model by
   design.
3. **Compare adapters only where they are comparable.** With trainable A, the input side of ΔW is
   seed noise, and full-matrix cosines between seeds are about 0.08 even for the same task.
4. **Use the target model's own mean adapter as a baseline.** In these experiments it beat every
   learned map by a wide margin.
5. **Use a graded measure next to task metrics.** Exact-match metrics hide partial transfer;
   gold-answer likelihood reveals it.
6. **Early stopping can stop hard tasks just before they take off.** Require a minimum number of
   passes over the data, and patience measured in passes rather than steps.
7. **Do not count format-only tasks as transfer successes.** Calibrate task difficulty against the
   few-shot model, and on the largest model that will be used.

## Reproduce

| Result | Config / script | Output |
|---|---|---|
| v2 dry run, trainable A | `configs/experiments/dev_a100_v2.yaml`, `scripts/multi_gpu.sh` | `results/dev_a100_v2*` |
| Frozen A vs trainable A | `scripts/frozen_a_test.sh` | `results/dev_3080_frozenA*`, `results/dev_a100_v2_cmp*` |
| Patience | `configs/experiments/dev_3080_patience.yaml` | `results/dev_3080_patience` |
| Task-count scaling | `scripts/scaling_test.sh`, `scripts/coverage_scaling.py` | `results/dev_3080_frozenA_scale` |
| Smart eyes | `scripts/eyes_test.sh` | `results/eyes_summary.txt` |
| Shared adapter | `configs/experiments/shared_adapter_d{64,128}.yaml` | `results/shared_adapter_d{64,128}` |

## Pending: zero-default shared adapter and the head-start test

`scripts/headstart_test.sh` runs the shared adapter with zero-initialised cores, so that an unseen
task gets no update by default. It then asks a weaker, practical question: given only 32, 128 or
512 examples of a held-out task on the target, does starting from the core learned on the other
model help the target learn faster than starting from zero or from the average training core?
Ordinary LoRA trained from scratch on the same examples is the reference.
