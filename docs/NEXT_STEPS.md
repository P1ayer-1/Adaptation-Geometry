# Next steps (2026-10-08)

The original question is answered at dev scale: a new task's adaptation does not carry from one
model to another, and a transferred adapter is not even a reliable head start. See
[`dev_scale_report.md`](dev_scale_report.md) for results and [`HANDOFF.md`](HANDOFF.md) for the
full state. What remains is a choice between closing this out and pivoting to what the results
turned up.

## Options

### 1. Is the "Universal Weight Subspace" an initialisation artefact? (recommended first)

A published claim says about 500 Mistral-7B LoRAs share one low-dimensional subspace. Our
findings suggest part of that could come from the initialisation:

- LoRA's A matrix stays 96-99.8% at its random initialisation.
- The same seed gives the identical random A on a base, and many community LoRAs use the default
  seed 42.

Plan:

- **Part 1 (laptop CPU, ~1 day, $0):** run the paper's subspace analysis on our own adapters in
  three regimes: shared seed (`dev_3080`, v1), per-task seed (`dev_a100_v2`), and frozen A
  (`dev_3080_frozenA`). If "universality" appears with shared seeds and vanishes with per-task
  seeds, that is the finding.
- **Part 2 (downloads, little or no GPU):** take public LoRAs for one base, regenerate PEFT's
  seed-42 initialisation, and measure how much of each A, and of the claimed subspace, is still
  that initialisation.
- **Same paper:** check whether continual-learning methods built on LoRA subspace angles (O-LoRA,
  C-LoRA, the 2026 forgetting law F = α(1 − cos²θ) + β) change their predictions under controlled
  seeding.

First, read the paper's methods: they may already control for seeds. Either outcome is
reportable. This is the most novel result available for the least cost.

### 2. Feed the findings into the RSI program (Lifespan night step)

Adaptations split into a generic component (the target's mean adapter carried 13-28% of the
benefit to unseen tasks) and stable task-specific parts. Proposal for the night step: distil only
the generic part into the base, and keep task-specific parts as a frozen-A adapter library chosen
by the System-1 router. This would live in the Lifespan repo, where grid036 v2 found replay beats
consolidating whole adapters.

Evaluator lessons to add to the RSI plan as amendments:

- a skill-not-format gate (gains must beat a few-shot baseline);
- take-off-aware kill rules (hard tasks sat at chance for ~6k examples, then learned; measure
  plateaus in exposure, not turns, and keep a late-take-off canary task);
- the target's mean adapter as the baseline for any transfer claim;
- graded likelihood next to pass/fail;
- carry skills across student generations as data, environments and verifiers, not adapters
  (this project's head-start result).

### 3. Steering vectors (activation-space transfer)

Published work reports that steering vectors transfer between models through a learned map
(arXiv 2503.04429, 2410.12877). Test whether behaviours the map never saw transfer, and whether
an adapter's activation shift transfers where its weights did not. Interesting, but effectively a
new project; it reuses the task panel and the evaluation code.

### 4. Stage 0 at 3B scale (not recommended now)

Roughly $30-50 on rented GPUs. Only worthwhile as a stronger version of the negative result for a
write-up; the dev results give little reason to expect the gate to pass.

## Recommended order

1. Merge this branch (PR opened 2026-10-08).
2. Option 1, part 1, on the laptop.
3. Depending on that: option 1, part 2 (a paper), or option 2 (Lifespan and the RSI plan).
