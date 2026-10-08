# Plan: make LoRA updates depend on the task, not the random A

*Written 2026-10-08 as a hand-off for a new session. Read [`HANDOFF.md`](HANDOFF.md) for the repo,
environment and rental practice, then this file.*

## Question

Can we change how LoRA is initialised or updated so that training the same task always converges
to the same update shape (ΔW = B·A), whatever the random seed, and so that similar tasks give
similar shapes?

## Why this matters

- With standard LoRA, A barely moves from its random start (4-22% here; median 23% in 500 public
  Mistral adapters). The update is built inside the random 16-dimensional input space A happened
  to start with. Two seeds of the same task agree at a global cosine of only **0.08**
  (`dev_scale_report.md`, finding 3).
- Freezing one A per model gives **0.92** agreement, but only because every adapter shares the same
  arbitrary input space. The shape is consistent, not task-determined.
- The universal-subspace check (`universal_subspace_check.md`) showed the published "universal"
  LoRA subspace is mostly a reused seed. A seed-free LoRA is the constructive answer. It would
  let adapter similarity measure tasks, which Lifespan and the RSI plan need for routing,
  deduplication and consolidation (`rsi_evaluator_amendments.md`, amendment 6).

## Methods to test (arms)

| Arm | Init of A | Update rule | Role |
|---|---|---|---|
| R | random, independent per task and seed (`init_seed_scope: task`) | standard AdamW | baseline (expect ~0.08) |
| F | one fixed random A per base, frozen (`configs/lora/r16_frozenA.yaml`) | B only | consistency without task-dependence (expect ~0.92 same task, high for *all* pairs) |
| G | **task gradient**: top-r right singular vectors of the full-weight gradient on the task | standard AdamW | new |
| P | random per task and seed | **preconditioned** (Riemannian / scaled GD) | new |
| GP | task gradient | preconditioned | new, if G and P each help |
| FT (optional) | full fine-tune of the target modules, then rank-16 truncated SVD of ΔW | - | reference "true shape" |

### G: task-gradient initialisation (LoRA-GA / LoRA-One style)

1. Before training, run K batches (start with K = 8, batch as in training) of the task's training
   data through the model, with the base weights requiring grad only for the target modules (or
   capture with hooks: grad_W = Σ δ_outᵀ x_in, which avoids materialising grads for the whole model).
2. For each target module, take G = ∂L/∂W (d_out × d_in) and its top-r right singular vectors V_r.
3. Set A = V_rᵀ, rescaled to the Frobenius norm PEFT's kaiming init would give (so learning rates
   stay comparable). Keep B = 0 so the model starts unchanged. (LoRA-GA also initialises B from the
   left singular vectors and subtracts the product from W. Treat that as an optional variant G2.)
4. **The K batches must depend on the run seed** (different samples or orders per seed).
   Otherwise seed agreement is trivially high and the test proves nothing. Record which examples
   were used.
5. Signs and order of singular vectors are arbitrary. This is harmless for ΔW comparisons, but
   fix a sign convention for reproducibility.

### P: preconditioned updates (Zhang & Pilanci 2024, "Riemannian Preconditioned LoRA")

After `backward()` and before the optimiser step, for each LoRA module:

- grad_A ← (BᵀB + δI)⁻¹ · grad_A
- grad_B ← grad_B · (AAᵀ + δI)⁻¹

with r × r matrices, so this is cheap. δ matters because B = 0 at the start (BᵀB is singular).
Start with δ = 1e-6 relative to the trace, check the paper's value, and sweep a little if training
is unstable. The paper pairs this with AdamW (preconditioning before Adam's moments) and also
reports plain-SGD results. Try AdamW first and keep the same learning rate schedule as arm R,
retuning only if it fails to learn.

### FT: reference shape (optional, only if G or P look promising)

Full fine-tuning of Qwen-0.5B's target modules may not fit the 8 GB laptop. Use a rental, or
restrict to attention projections. Truncate each ΔW to rank 16 by SVD and compare the arms against it.

## Tasks and comparison levels

Base: **Qwen2.5-0.5B** first (`configs/bases/dev_qwen2.5-0.5b.yaml`). Add Llama-3.2-1B only if the
result is positive.

Tasks (v2 panel, `configs/tasks/v2/`): T3 paraphrase, T7 python, T8 arithmetic, T9 clinical
(learned reliably), plus T5 concise / T6 verbose and T2 NLI / T3 paraphrase as candidate
"related" pairs.

Measure similarity at four levels, from most to least similar:

1. **Same task, same data, different seed.** Main test.
2. **Same task, disjoint data halves** ("same topic, different examples"). Needs a small feature: a
   data offset / disjoint-subset option next to `max_train_examples` in `train_lora.py`.
3. **Related tasks** (T5/T6, T2/T3).
4. **Unrelated tasks** (e.g. T7 vs T9).

A method succeeds if level 1 rises well above the R baseline and levels stay ordered 1 > 2 > 3 > 4
with clear gaps. Arm F is the warning case: high similarity everywhere means consistency without
meaning.

## Measures

- `uag seed-compare` (`diagnostics.py`): global ΔW cosine and subspace overlaps for same task /
  different seeds, different tasks / same seed, different tasks / different seeds. Extend the
  groups to levels 2-3 above.
- Add **gauge-free** comparisons: overlap of the top-16 input-side and output-side singular
  subspaces of ΔW (mean cos² of principal angles), next to the global cosine.
- Factor movement (already logged per run): does A move more under G and P?
- Task scores against the few-shot baseline. **A method that does not learn the task as well as
  arm R (within ~0.03) fails, however consistent it is.**
- Training speed (steps to take-off): LoRA-GA / LoRA-One report faster convergence. The late
  take-off tasks (T1, T2) are a useful extra check if time allows.

## Run size and order

1. Implement G (config `lora.a_init: gradient`; today `LoraSettings.validate` allows only
   random/whitened/pca and requires `init_seed_scope: base` for non-random, so relax that for
   `gradient`). The data-informed init hook point is where `apply_eyes` is called in
   `train_lora.py` (~line 216); `eyes.py` shows the forward-hook pattern for collecting
   per-module statistics. Add unit tests on the tiny models (`configs/bases/tiny_*`), including
   one that two seeds draw different gradient batches.
2. Implement P as a gradient transform applied between `backward()` and `optimizer.step()`, behind
   a config flag (e.g. `lora.precondition: none|riemannian`, `lora.precondition_delta`). Remember
   that new config fields must keep old run hashes (`check_reusable`, `pipeline._LATE_GATE_FIELDS`
   pattern: omit at defaults).
3. Pilot: arms R, G, P × tasks T3, T8 × seeds 0, 1, 2 on Qwen-0.5B = 18 runs. Run `uag benchmark`
   first to estimate time. Use early stopping with `min_epochs >= 2` (`HANDOFF.md`, pitfalls).
4. If G or P lifts same-task agreement clearly: full panel (add T5, T6, T7, T9, T2), disjoint-half
   runs, GP, and then FT as the reference.

Commit in small steps with `pytest` passing before each commit; push the branch.

## Hardware

Needs CUDA. The workstation (`ssh noahp@workstation`, Windows, AMD RX 580) is CPU-only for this.
Options: the laptop RTX 3080 (fits Qwen-0.5B but flaky on long runs, see `HANDOFF.md`) or a
rented GPU (preferred for the pilot and anything with Llama). Never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` under WSL.

## References (verify before citing)

- LoRA-GA: Wang et al. 2024, "LoRA-GA: Low-Rank Adaptation with Gradient Approximation", arXiv 2407.05000.
- LoRA-One: Zhang et al. 2025, one-step full gradient suffices for LoRA, arXiv 2502.01235.
- Riemannian Preconditioned LoRA: Zhang & Pilanci 2024, arXiv 2402.02347.
- PiSSA (SVD of W, task-free; the same init for every task, so it is closer to arm F): arXiv 2404.02948.

## Expected outcomes

- **G works** (same-task agreement ≫ 0.08, ordered levels, no accuracy loss): task-determined
  adapters are cheap. Write it up as the constructive half of the universal-subspace note and
  propose it for Lifespan's adapter library.
- **P works but G does not:** the shape is reachable, but only through the training dynamics.
  Check how many passes it needs.
- **Neither works:** LoRA shapes are intrinsically non-unique at rank 16, and adapter geometry
  should not be used for task similarity. Use activation or behaviour comparisons instead. That is
  also worth stating in the note.
