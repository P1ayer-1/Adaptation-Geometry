# Universal Adaptation Geometry

**Do task adaptations have a stable, compositional geometry across language-model bases?**

Given base model *m* and transformation *t*, let Δ(m,t) be the effective LoRA update `scale·BA`.
This project tests whether a lower-dimensional task representation *z_t* and model-specific
decoders *D_m* exist such that `Δ(m,t) ≈ D_m(z_t)`, with *z_t* staying useful for tasks and
models that were not used to fit *D*. The cheapest decisive test is Stage 0: learn a
source→target map on eight transformations and check whether it predicts useful target
updates for the two held-out ones.

This repository implements the spec's staged plan: Stage 0 end to end, plus the building
blocks for Stages 1–3.

| Stage | What | Where | Status |
|---|---|---|---|
| 0 | 4 bases × 10 transformations × 3 seeds, held-out transfer, predeclared gate | `pipeline.py`, `transfer.py`, `analysis.py`, `configs/experiments/stage0.yaml` | Implemented; validated on tiny CPU bases; needs pinned revisions + GPU to launch |
| 1 | Seed / dataset / rank invariance, layer localisation, controls | `geometry.py`, `spectral.py` | Library functions + tests |
| 2 | Shared latent *z_t* + thin per-model decoders, LOTO / LOMO | `latent_model.py` | Linear + bilinear decoders + tests |
| 3 | Latent arithmetic vs LoRA composition baselines | `composition.py`, `configs/tasks/composition/` | Library functions + one factorial task |
| 4 | Routing / mixture of adaptations | — | Deliberately not started (spec §9) |

## Install

```bash
pip install -r requirements.lock      # exact validated versions
pip install -e ".[dev]"
pytest                                # ~50 tests, ~2 min on CPU, no hub access needed
```

## Quickstart: smoke test (CPU, about 4 minutes)

```bash
bash scripts/smoke_test.sh            # 2 tiny bases × 2 tasks × 2 seeds + toy held-out maps
cat results/smoke_decision.md
```

The smoke test builds randomly initialised tiny Llama/Qwen2 bases with a byte-level tokenizer,
then trains, verifies ΔW, evaluates, fits every map type, predicts held-out updates, evaluates
them, and writes the decision memo. **Its numbers carry no scientific weight.** Random frozen
tiny bases cannot learn the semantic tasks at CPU scale, so every cell is correctly marked
ineligible and the memo reads NO-GO. The gate and RecoveredLift logic are tested separately on
synthetic results (`tests/test_analysis.py`).

## Dry run on a home GPU (before renting)

Run the pipeline once on real small models first (Qwen2.5-0.5B + Llama-3.2-1B):

```bash
huggingface-cli login                                     # and accept the Llama-3.2 licence
bash scripts/dry_run.sh                                   # v2 panel, 3 tasks x 2 seeds, 8 GB GPU
bash scripts/dry_run.sh configs/experiments/dev_a100_v2.yaml   # all 10 tasks x 2 seeds, rented GPU
```

Both are resumable (rerun after a crash). The script ends with `uag dry-run-report`, saved to
`results/<config>.dry_run_report.txt`. The report covers:

1. Training finished (no OOM or divergence), peak VRAM, and where each run stopped.
2. ΔW reconstruction check.
3. Headroom. Direct LoRA must beat the **few-shot** base, no base may be at ceiling (≥ 0.9),
   and the lift must not be format-dominated.
4. Token budget from the learning curves.
5. Geometry sanity: factor movement and seed agreement.
6. Timing on this machine.

The v1 dry run (`dev_3080`, kept as a record) is why the panel changed. With 5 worked examples
in the prompt, the untrained bases already scored 1.00 on JSON and 0.98–1.00 on arithmetic.
The LoRA lift there was output format, not skill.

## Baselines: zero-shot vs few-shot

Every base is evaluated twice on every task:

- **Zero-shot**: the bare instruction.
- **Few-shot**: `fewshot.k` (default 5) deterministic training examples in the prompt.
  Classification shots cycle through the labels. For generation, output is cut where the model
  starts a new `Input:` block, and at the first line break when every shown answer is one line.

The declared `gate.baseline` (default `fewshot`) is the S_base used for lift, eligibility and
RecoveredLift, so an adapter only gets credit for what demonstrations cannot give. The
zero-shot base is still reported. The memo and dry-run report flag two kinds of cells:

- **ceiling**: a base scores ≥ 0.9.
- **format**: the few-shot gain is ≥ 50% of the zero-shot-referenced lift.

## Token budget (identical across bases)

Training stops early on the validation task metric: `early_stopping_patience` validations
without a gain above `early_stopping_min_delta`, never before `min_steps`, and never beyond the
task's token cap. Each run manifest records the stop step, the reason and the selected step.
Caps are set from data:

1. Run a pilot (or `dev_a100_v2`) with a generous cap.
2. Run `uag learning-curves -e <config>`. For every run it reports `t95`, the tokens needed to
   reach 95% of the run's total improvement. It then proposes a cap per task: 2 × the largest
   `t95` over bases and seeds, rounded up.
3. Put the proposals in `configs/train/stage0.yaml` under `max_tokens_by_task` before declaring
   the gate.

Every base gets the same cap and stopping rule for a task, so budgets stay identical across
bases. A run still improving at its cap is flagged: raise that cap.

## Is ΔW task signal or random-init noise?

LoRA starts with a random A and B = 0. In the v1 dry run, A moved only 7–16% from its init.
A's input directions still overlapped the init at 0.98–0.99 (chance is about 0.01). Different
tasks trained with the same seed shared input directions at 0.97–0.99, because they shared the
same random A₀. Two changes follow:

- `lora.init_seed_scope: task` (now in `r16.yaml`) gives every (task, seed) its own init.
- Every manifest records factor movement: ‖A − A₀‖/‖A₀‖, ‖B‖, and row-space overlap with A₀.
  `uag seed-compare -e <config>` compares same-task/different-seed,
  different-task/same-seed and different-task/different-seed ΔW similarity against chance.

`configs/experiments/dev_3080_seed2.yaml` adds a second v1 seed for T4/T8 on top of the
finished dry run.

## Dev-scale findings so far (Qwen2.5-0.5B + Llama-3.2-1B; not Stage-0 evidence)

| Question | Result | Where |
|---|---|---|
| Does the v1 panel measure skill? | No: 5-shot bases already scored 0.90-1.00 | `results/dev_3080*` |
| Does trained LoRA A move? | Barely: 4-22%; input side of ΔW stays at its random init | `results/dev_a100_v2*` |
| Does held-out transfer work with trainable A? | No: a held-out task's input side lies in the training span at chance; maps predict ~0 | same |
| Does a frozen A per base still learn? | Yes: within 0.03 of trainable A on every cell | `results/dev_3080_frozenA*` |
| Are frozen-A adapters reproducible across seeds? | Yes: same-task cosine 0.92-0.93 (trainable: 0.08) | same |
| Does held-out transfer work with frozen A? | Not on task metrics. Gold-answer likelihood: maps recover 1-6%, random ~0, the target's own mean adapter up to 21-28% | `uag graded-transfer`, `results/*/graded_transfer.json` |
| Why so little? | A held-out task writes mostly to its own output directions: 6-16% (k=16) / 12-30% (k=64) inside the training tasks' span, vs 94-96% for a second seed of the same task | |
| Do T1/T2 learn with patient stopping? | Yes: all four failed dry-run runs reach 0.92-1.00 after ~6k examples at chance | `results/dev_3080_patience` |

## Renting a GPU

- **Price a card first.** `bash scripts/benchmark_gpu.sh <$/h>` trains one Stage-0-sized run
  of Llama-3.2-3B for up to 45 min (`configs/train/gpu_24_48gb.yaml`: micro-batch 8, no
  gradient checkpointing). It then verifies ΔW, evaluates, and prints the cost per run and a
  Stage-0 projection. Run it on each candidate card (e.g. A40 vs H100) and compare.
- **Several GPUs in one machine** (e.g. 4× RTX 5090): `bash scripts/multi_gpu.sh <config> <n_gpus>`
  deals the tasks round-robin to the GPUs (one process per GPU) and runs the same steps as
  `dry_run.sh`. Keep every run of one experiment on the same GPU type.
- **Manual split.** `train`, `eval`, `fit-maps` and `learning-curves` accept `--base`/`--task`.
  For `eval --stage transfer` and `fit-maps`, `--base` means the *target* base; fitting still
  uses every training task. For example:
  ```bash
  CUDA_VISIBLE_DEVICES=0 uag train -e configs/experiments/stage0.yaml --base A1_qwen2.5-1.5b B_llama3.2-3b
  CUDA_VISIBLE_DEVICES=1 uag train -e configs/experiments/stage0.yaml --base A2_qwen2.5-3b C_gemma2-2b
  ```
- Finished runs are only reused when their dataset version and LoRA/training settings match
  the config. Otherwise the pipeline stops and asks for a new experiment name.

## Launching Stage 0

1. **Pin the bases.** Candidate configs live in `configs/bases/stage0_*.yaml` (Qwen2.5-1.5B/3B
   as the A1/A2 family pair, Llama-3.2-3B as B, Gemma-2-2B as C). The Stage-0 config refuses to
   load until every hub revision is a 40-character commit hash:
   ```bash
   uag pin-revisions configs/bases/stage0_*.yaml
   ```
2. **Pilot the task panel** (spec §17, days 3–5). Use one seed, evaluate on the validation
   split only, and confirm each base has headroom on each task:
   ```bash
   bash scripts/stage0_train_grid.sh configs/experiments/pilot.yaml
   uag analyze -e configs/experiments/pilot.yaml   # headroom table
   ```
   Replace any task where a base sits at ceiling or floor before freezing the panel.
3. **Run the grid, fit maps, evaluate, decide:**
   ```bash
   bash scripts/stage0_train_grid.sh    # 120 LoRA runs + raw-base and direct-LoRA evals
   bash scripts/stage0_fit_maps.sh      # freezes the gate, fits maps, predicts held-out updates
   bash scripts/stage0_eval.sh          # evaluates predictions, writes results/stage0_decision.md
   ```
   Every stage is idempotent: completed runs and evaluations are reused, never silently redone.

## How the protocol is enforced in code

| Requirement (spec) | Mechanism |
|---|---|
| Immutable splits + hashes (§10.3) | `data.py` writes SHA-256 manifests (`data/manifests/`, committed) and re-verifies them on every load. Test inputs are generated first and excluded from valid/train. Regenerating a version with different content raises an error. |
| Pinned bases (§5.1) | `stage0.yaml` sets `require_pinned_revisions: true`; loading fails otherwise. Tiny/local bases record a weight hash. |
| Frozen base, reload into exact base (§20.2) | A weight fingerprint is taken before and after training; `load_trained` refuses mismatched revisions or weights. |
| Never compare raw A/B (§13, §18) | `extract_delta.py` stores every update as its exact thin SVD of `scale·BA`, computed by QR without forming the dense matrix. |
| Δ reconstruction = PEFT forward (§13) | `verify_delta_reconstruction` checks each module against PEFT's `get_delta_weight` and compares full-model logits (base + ΔW vs. active LoRA). This runs for every trained adapter. |
| Semantic, explicit module matching (§13) | `alignment.py` holds per-architecture tables mapping to canonical `q,k,v,o,up,down,gate`. Missing or fused projections (Phi's absent gate, Phi-3's `qkv_proj`) are recorded as omissions, never reshaped. |
| Held-out IDs cannot enter map fitting (§13) | `TaskSplit.check_fit_inputs` raises `HoldoutLeakError` at the `fit_pair_map` / latent-model boundary. The pipeline never loads held-out target runs while fitting. Normalisation statistics come from training tasks only. |
| Low-capacity maps (§5.4) | Procrustes and bilinear maps in truncated-SVD coordinates; same-shape two-sided Procrustes acts as the identity outside the training span. |
| Baselines (§5.4) | Naïve copy (only where shapes match), Cross-LoRA-*style* base-weight subspace projection, norm-matched random low-rank update. Direct LoRA is the upper reference. |
| Skill vs format | Few-shot base evaluation (deterministic shots from train, recorded) is the declared lift baseline; ceiling and format-dominated cells are flagged. |
| No stale reuse | A finished run or evaluation is reused only if its dataset version and LoRA / training settings match the config. |
| Predeclared gate (§5.6) | `uag declare-gate` hashes the gate, splits and few-shot settings before held-out evaluation; later edits are refused. The gate is evaluated for a predeclared `primary_method`, with no post-hoc choice of method. |
| Statistics (§15) | Example-level bootstrap CIs, seed-level variation reported separately, paired bootstrap over identical source×target×task cells, direct-lift eligibility threshold. |
| Failure / exclusion log (§21) | `results/<experiment>/exclusions.jsonl` records divergence, failed Δ checks, non-applicable methods and missing runs. The memo lists them. |
| Robustness + collateral (§5.5, §10.3) | `alt_templates: [v1-alt]` re-evaluates everything under an alternate prompt format; `control_tasks` measures damage to unrelated tasks. |

## Transformations (Stage-0 panel v2)

All ten tasks are generated deterministically by rules (`src/uag/tasks_v2.py`; v1 is in
`tasks.py`), so no target is defined by another model's preferences. Every gold target scores
1.0 under its own metric (tested).

The v1 panel (`configs/tasks/*.yaml`, manifests `*.v1.yaml`) is kept unchanged. The v2 panel
(`configs/tasks/v2/`, manifests `*.v2.yaml`) was built so that five demonstrations can't reveal
the rule. The last two columns are base scores on 100 validation examples (Qwen2.5-0.5B /
Llama-3.2-1B), zero-shot and 5-shot:

| ID | Transformation (v2) | Metric | Zero-shot | 5-shot |
|---|---|---|---|---|
| T1 | Hidden-rule triage: keep/return = polarity XOR hidden split of 40 products (replaces sentiment) | accuracy (chance 0.5) | 0.45 / 0.55 | 0.49 / 0.60 |
| T2 | Relational NLI: transitive comparisons over two chains | accuracy (chance 0.33) | 0.39 / 0.39 | 0.37 / 0.35 |
| T3 | Paraphrase: role swaps, voice, number words, before/after | accuracy (chance 0.5) | 0.64 / 0.43 | 0.61 / 0.55 |
| T4 | Nested JSON + distractor + hidden sector codes + ISO languages | exact record | 0.00 / 0.00 | 0.08 / 0.29 |
| T5 | Concise answer to a context question | first answer correct ∧ ≤ 4 words | 0.00 / 0.00 | 0.27 / 0.35 |
| T6 | Verbose answer to a context question | first answer correct ∧ 25–80 words | 0.00 / 0.25 | 0.22 / 0.31 |
| T7 | Python: 2–3 chained list steps + aggregate, or string pipeline | unit tests | 0.02 / 0.06 | 0.05 / 0.59 |
| T8 | 3–5 step arithmetic, 3-digit numbers, distractor | exact number | 0.00 / 0.10 | 0.11 / 0.14 |
| T9 | Compositional clinical morphology (organ root × suffix) | exact match | 0.00 / 0.00 | 0.01 / 0.14 |
| T10 | Sort, number, case by length, `TOTAL:` footer | exact lines | 0.00 / 0.00 | 0.00 / 0.00 |
| C1 | Arithmetic + concise (Stage 3, v1) | both factors | | |

T7 is the weakest. Llama-1B reaches 0.59 from demonstrations alone, and larger Stage-0 bases
may be higher, so check it in the pilot. T10 is at 0 even with shots; the dry run will show
whether LoRA can learn it at all.

Instructions are deliberately minimal ("Answer the question."), so that style and format
behaviours live in the adapter rather than in the prompt.

## Repository layout

```
configs/      bases/ tasks/ lora/ train/ experiments/     one YAML schema (src/uag/config.py)
data/         manifests/ (committed hashes)  generated/ (gitignored, regenerable)
src/uag/      config, tasks, data, prompts, provenance, models, tiny,
              train_lora, extract_delta, spectral, evaluate, metrics,
              alignment, transfer, pipeline, analysis, geometry,
              latent_model, composition, cli
scripts/      stage0_train_grid.sh  stage0_fit_maps.sh  stage0_eval.sh  smoke_test.sh
artifacts/    adapters, maps, predictions, cached spectra (gitignored; see artifacts/README.md)
results/      raw/ (per-example JSONL, gitignored)  tables/  <experiment>_decision.md
tests/        acceptance tests for the spec's §20 work order + synthetic-geometry tests
```

## Known limitations / next steps

- `cross_lora` is a data-free *approximation* of Cross-LoRA's subspace alignment. It projects
  through rank-ordered, sign-canonicalised base-weight singular vectors. Before citing
  comparisons, replace or check it against the reference implementation.
- The PorTAL baseline (spec §7) is not reproduced here. Stage 2 should start by running
  `portallib`'s released recipe.
- The latent model's capacity sweep covers linear and bilinear decoders; the small
  MLP/hypernetwork decoder is not implemented yet.
- LoRA-Soups-style mixing uses a global weight grid chosen on validation data; learned
  per-module (CAT-style) weights are not implemented.
- rsLoRA is supported; DoRA and `rank_pattern` adapters are rejected explicitly because they
  are not of the form `scale·BA`.
- T7 executes model-generated code in an isolated subprocess with a timeout. That is not a
  security sandbox, so evaluate untrusted models in a disposable container.

## Literature anchors

PorTAL (Ramp Labs, 2026) and `portallib`; Cross-LoRA (Xia et al., 2025, arXiv:2508.05232);
LoRA Soups (Prabhakar et al., 2024, arXiv:2410.13025); LoRA vs Full Fine-tuning: An Illusion
of Equivalence (Shuttleworth et al., 2024, arXiv:2410.21228). This project does not claim to
invent portable LoRAs. Its contribution is an empirical study of the latent geometry itself.
