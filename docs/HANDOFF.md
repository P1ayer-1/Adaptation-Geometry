# Handoff: Universal Adaptation Geometry

*Written 2026-10-04 at the end of a long working session (head-start results added the same evening). Read this first, then
[`dev_scale_report.md`](dev_scale_report.md) for the results with tables.*

## Where things stand (one paragraph)

The project tested whether LoRA task adaptations share a cross-model geometry, so that a
**new** task learned on one model can be carried to another without target-task data. On
dev-scale models (Qwen2.5-0.5B, Llama-3.2-1B) and a 10-task rule-generated panel, the answer
for this strong form is **no**. Three independent methods agree: weight-space maps, frozen-A
maps and a shared adapter with per-model connectors all recover only about 0.4-6% of the
benefit of training the target directly. Within one model, adapters are highly reproducible
once the LoRA A matrix is frozen per model (seed agreement 0.92 against 0.08). Stage 0 (the
expensive 3B-scale grid) has **not** been run, and the dev results give little reason to
expect its gate to pass.

## Repository state

- Branch: `claude/task-design-fixes` (about 40 commits ahead of `main`; no PR opened; all pushed).
- Tests: `pytest`, about 60 tests, a few minutes on CPU. All passing at the last commit.
- Environment: micromamba env `lora_research` (Python 3.12, torch 2.14 cu130) on the user's
  RTX 3080 laptop under WSL2. Use `~/micromamba/envs/lora_research/bin/{python,pytest,uag}`.
- Data: dataset manifests are committed (`data/manifests/*.v1.yaml`, `*.v2.yaml`); generated data
  is regenerable and gitignored. v1 tasks are kept unchanged; v2 (`configs/tasks/v2/`) is the
  current panel.
- Artifacts (adapters, maps) are gitignored. Local copies exist for `dev_a100_v2`,
  `dev_3080_frozenA`, `dev_3080_frozenA_scale`, `dev_3080_patience`, `eyes_*` (manifests and
  logs only) and `shared_adapter_d*` (cores only; connector weights were not downloaded).
  Raw archives from rentals: `D:\Swap\dry_run_v2.tgz` (3 GB, includes adapters).

## What was built this session (beyond the original Stage-0 pipeline)

| Feature | Where | Why |
|---|---|---|
| Few-shot baseline (lift measured against k=5 worked examples), ceiling and format flags | `evaluate.py`, `analysis.py`, `report.py` | v1 lift was output format, not skill |
| Task panel v2 (harder, hidden rules) | `tasks_v2.py`, `configs/tasks/v2/` | 5-shot bases were at 0.9-1.0 on v1 |
| Factor movement, seed comparison | `train_lora.py`, `diagnostics.py` (`uag seed-compare`) | LoRA A barely moves; shared-seed confound |
| `lora.init_seed_scope` (seed / task / base), `lora.train_A` (frozen A) | `config.py`, `train_lora.py` | independent inits; one fixed A per base |
| Smart eyes (`lora.a_init` whitened / pca) | `eyes.py` | tested; not needed |
| Early stopping, `min_epochs`, per-task token caps, `uag learning-curves` | `train_lora.py`, `budget.py` | hard tasks take off late |
| `max_train_examples`, `max_steps` | `train_lora.py` | few-example experiments |
| Stale-run guard | `train_lora.check_reusable` | never mix runs from different settings |
| Transfer baselines `random_coords`, `target_mean`; graded transfer (`uag graded-transfer`, RecoveredNLL); gate criterion `graded_beats_baselines` | `transfer.py`, `graded.py`, `analysis.py` | target_mean beats every learned map |
| Shared adapter (connectors + shared cores; `core_init` identity/zero; resumable phase 1; head-start phase) | `shared_adapter.py` (`uag shared-adapter`) | function-space version of the hypothesis |
| GPU tooling: `uag benchmark`, `scripts/multi_gpu.sh`, `scripts/keep_running.sh`, CUDA-failure fast exit | `benchmark.py`, `cli.py`, `scripts/` | rentals and the flaky laptop GPU |

Stage-0 configs were updated but **not run**: frozen A (`configs/lora/r16_frozenA.yaml`), v2 panel,
target_mean and random_coords baselines, the graded criterion, early stopping with two passes
minimum (`configs/train/stage0.yaml`). Caps in that file are placeholders.

## Results (short; full tables in the report)

1. v1 panel: format, not skill. v2 panel: 5-shot 0.00-0.60, direct LoRA 0.90-1.00.
2. Trainable A moves 4-22%; its input side is seed noise. Shared seeds manufacture 97-99%
   cross-task overlap.
3. Weight-space maps predict ~0 updates (held-out input side lies in the training span at chance).
4. Frozen A: same learning, seed agreement 0.92-0.93; held-out output side only 6-16% inside the
   training span; maps recover 1-6%, the target's own mean adapter up to 21-28%.
5. Coverage grows ~2-4% per added training task, with no sign of levelling off; maps 2.3% -> 3.2% from 2 to 8 tasks.
6. T1/T2 sit at chance for ~6k examples, then take off; frozen A takes off at 0.8-1.1 passes.
   Smart eyes give no consistent gain.
7. Shared adapter: expressive (a new core through frozen connectors reaches 0.79-1.00) but not
   portable (0.4-5% of the gap closed in the other model). Identity-initialised cores carry a
   harmful default; zero-initialised cores remove it, but the transferred core still hurts.
9. Head start: a transferred core does not reliably help the target learn from 32-512 examples,
   and plain LoRA beats the shared-core route on T8 (see below).
8. bf16 merging does not distort evaluation.

## Head-start test (last experiment of the session)

Question: when the target gets only N = 32 / 128 / 512 examples of a held-out task, does
starting the core from the one learned on the other model help, compared with zero, the mean
training core, and ordinary LoRA from scratch? Run with `scripts/headstart_test.sh` on 2x A100
(GPU 0 was partly occupied by another tenant, so everything ran on GPU 1 via `SHARED_GPU=1
LORA_GPU=1 JOB_GPUS="1 1 1 1"`). Results: `results/shared_adapter_zero_d64_gpu/`,
`results/headstart_summary.txt`.

**Zero-default transfer (no target training).** A zero core is now exactly the untouched model,
so the identity-default flaw is gone. The transferred core still does not help: it makes the gold
answer *less* likely than doing nothing (RecoveredNLL −0.04 and −0.35 on T4, −4.0 and −4.8 on T8),
about as harmful as a norm-matched random core. The mean training core helps slightly on 3 of 4
cells (up to 0.21). Ceilings: T4 0.995 on both models; T8 only 0.47 / 0.68, lower than with
identity cores (0.89 / 0.87).

**Head start (test score after training on the target from N examples):**

| Task, target | N | Zero start | Mean core | Transferred core | Plain LoRA |
|---|---|---|---|---|---|
| T4, Llama | 32 | 0.565 | 0.600 | **0.705** | 0.385 |
| T4, Llama | 128 / 512 | 1.00 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| T4, Qwen | 32 | **0.625** | 0.250 | 0.185 | 0.195 |
| T4, Qwen | 128 / 512 | 0.99 / 0.99 | 0.98 / 1.00 | 1.00 / 1.00 | 0.93 / 1.00 |
| T8, Llama | 32 / 128 / 512 | 0.22 / 0.46 / 0.53 | 0.38 / 0.50 / 0.51 | 0.00 / 0.21 / 0.41 | **0.73 / 0.80 / 0.91** |
| T8, Qwen | 32 / 128 / 512 | 0.11 / 0.23 / 0.34 | 0.09 / 0.23 / 0.43 | 0.01 / 0.11 / 0.26 | **0.67 / 0.85 / 0.93** |

Verdict:
- **No reliable head start.** The transferred core helped in 1 of 12 settings (Llama T4 at
  N = 32), hurt in 7 (every T8 setting and Qwen T4 at N = 32) and tied in the rest. The one win
  does not hold in the other direction.
- **Plain LoRA from scratch beats the whole shared-core route on T8 by a wide margin** (0.67-0.93
  against at most 0.53) and matches it on T4. Learning only a core through frozen connectors is a
  weaker way to learn a new task than ordinary LoRA, so the connector basis limits learning as well
  as failing to transfer.
- For the RSI plan: carrying skills to a new student as adapters does not pay off at this scale;
  carry them as data, environments and verifiers.

## Open threads, ranked

1. **Is the "Universal Weight Subspace" an initialisation artefact?** A published claim says ~500
   Mistral-7B LoRAs share a low-dimensional subspace. Our data: LoRA A stays at its random init,
   and identical seeds give identical inits (many public LoRAs use the default seed 42). Tests:
   (a) CPU-only, on our adapters: run the paper's subspace analysis under shared-seed (v1
   `dev_3080`), per-task (`dev_a100_v2`) and frozen-A runs; (b) download public LoRAs for one
   base, regenerate PEFT's seed-42 init, measure how much of A (and of the claimed subspace) is
   init. Check the paper's methods first. Most novel, cheapest.
2. **LoRA-geometry continual-learning methods may measure the seed.** O-LoRA, C-LoRA and a 2026
   forgetting law (F = α(1 − cos²θ) + β, arXiv 2603.02224) use LoRA subspace angles. Recompute
   with shared vs independent seeds and frozen A.
3. **"Consolidate the shared part, keep the specific part"** for the RSI plan's night step.
   Adaptation = generic component (target_mean) + stable task-specific component. Distil only the
   generic part into the base; keep specific parts as a frozen-A B-matrix library selected by a
   calibrated System-1 router. Directly addresses Lifespan's grid036 finding that consolidating
   whole adapters hurts new-phase learning.
4. **Activation-space version (steering vectors).** Cross-model transfer of steering vectors is
   reported (arXiv 2503.04429, 2410.12877). Test held-out behaviours through a map learned on
   generic text, and adapter → activation shift → map → other model.
5. **Stage 0 at 3B scale.** Only worth it as a stronger negative result, or after a positive
   result in 1, 3 or 4. Upper-bound cost ~$30-50 (benchmark: $0.41-0.43 per run).

## Connections to the user's other projects

- **Lifespan** (`D:\Lifespan\Lifespan`): sleep consolidation of LoRAs into a 30M model.
  grid036 v2 found replay (arm B) beats consolidation (arm D); LoRA did not protect old phases;
  frozen embeddings were the bottleneck. Thread 3 above builds on this.
- **RSI program plan** (Oct 4, pasted in chat; three loops: harness, System 1, weights; evaluator
  "Holdout"): lessons to propose as amendments: a skill-not-format gate (few-shot baseline);
  take-off-aware kill rules (plateaus in exposure, not turns; keep a late-take-off canary task);
  target_mean as baseline for any transfer claim; graded likelihood next to pass/fail; adapters do
  not port across student generations (stage 3→4→5), so carry skills as data / environments /
  verifiers unless the head-start test says otherwise.
- **Earlier chat on gradient descent** (hypernetworks, Doc-to-LoRA, universal subspace,
  test-time training, SEAL, Absolute Zero): thread 1 is the direct link; "freeze a basis, learn
  coefficients" works within a model (our shared adapter) but not across models.

## How to run things

- Local smoke: `bash scripts/smoke_test.sh` (CPU, minutes).
- Rental setup (used three times): clone the branch into `/workspace`, `uv pip install -r
  requirements.lock && uv pip install -e ".[dev]"` in `/venv/main`, put the HF token in
  `/workspace/.hf_home/token`, prefetch both models, launch scripts with `setsid nohup ... &
  < /dev/null` so they survive disconnects. If `torch.cuda.is_available()` is False (old driver),
  install `torch==2.14.0 --index-url https://download.pytorch.org/whl/cu128`.
- Reaching a rental from WSL: the SSH key lives at `C:\Users\noahp\.ssh\id_ed25519` (permissions
  too open for ssh under WSL). Load it into a temporary agent instead of copying it:
  `ssh-agent -a <sock>; SSH_AUTH_SOCK=<sock> ssh-add - < /mnt/c/Users/noahp/.ssh/id_ed25519`.
  Remove it afterwards with `ssh-add -D`.
- Shared adapter: `uag shared-adapter -e <config> --phase all|connectors|heldout|evaluate|headstart`.

## Pitfalls that cost time (do not repeat)

- **`pkill -f <pattern>` killed my own shell four times**, because the pattern appeared in the
  invoking command line. Kill by PID or by process group (`ps -eo pgid,cmd | grep "[x]yz"`), or
  put the kill in a separate script file.
- **Laptop GPU (RTX 3080, 8 GB, WSL2, also drives the display) is flaky** for long runs:
  "CUDA error: unknown error", cuBLAS failures, OOM. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
  breaks CUDA memory mapping under WSL; never set it. The shared adapter with two models does not
  fit usefully on 8 GB (~14-52 s/step). Rent for anything with two models.
- **Early stopping**: patience in steps stops hard tasks before take-off; use `min_epochs >= 2`.
- **Rental surprises**: a 5090 host had a driver too old for cu130 PyTorch; a 2x A100 host had
  30 GB of GPU 0 held by an invisible foreign process. Check `nvidia-smi` memory before launching.
- **Gate declarations are hashed**: any change to gate settings needs a new experiment name; new
  gate fields must be omitted at their defaults (`pipeline._LATE_GATE_FIELDS`) to keep old hashes.
- **Gradient checkpointing with forward hooks**: run `backward()` inside the context that enables
  the hooks, or recomputation mismatches.
