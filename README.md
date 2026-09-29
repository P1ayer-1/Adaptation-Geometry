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
pytest                                # 34 tests, ~1 min on CPU, no hub access needed
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

The code has only been exercised on tiny random models. Run it once on real small models first
(Qwen2.5-0.5B + Llama-3.2-1B, 3 tasks, 1 seed; sized for an 8 GB GPU such as an RTX 3080):

```bash
huggingface-cli login            # and accept the Llama-3.2 licence on its model page
bash scripts/dry_run.sh          # ~1-3 h; resumable, rerun after any crash
```

It ends with `uag dry-run-report`, which answers the four pre-rental questions as PASS/CHECK
(no OOM, ΔW check passes, direct LoRA improves each task, time per run / per eval example)
and saves the report to `results/dev_3080.yaml.dry_run_report.txt`. If you run out of memory,
lower `max_seq_len` or `batch_size` (raise `grad_accum`) in `configs/train/dev_3080.yaml`.

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
| Predeclared gate (§5.6) | `uag declare-gate` hashes the gate + splits before held-out evaluation; later edits are refused. The gate is evaluated for a predeclared `primary_method`, with no post-hoc choice of method. |
| Statistics (§15) | Example-level bootstrap CIs, seed-level variation reported separately, paired bootstrap over identical source×target×task cells, direct-lift eligibility threshold. |
| Failure / exclusion log (§21) | `results/<experiment>/exclusions.jsonl` records divergence, failed Δ checks, non-applicable methods and missing runs. The memo lists them. |
| Robustness + collateral (§5.5, §10.3) | `alt_templates: [v1-alt]` re-evaluates everything under an alternate prompt format; `control_tasks` measures damage to unrelated tasks. |

## Transformations (Stage-0 panel)

All ten are generated deterministically by rules (`src/uag/tasks.py`), so no target is defined by
another model's preferences. Every gold target scores 1.0 under its own metric (tested).

| ID | Transformation | Metric | Scoring |
|---|---|---|---|
| T1 | Sentiment | accuracy (+ macro-F1) | label log-likelihood |
| T2 | NLI (entail / neutral / contradict) | accuracy (+ macro-F1) | label log-likelihood |
| T3 | Paraphrase equivalence | accuracy (+ macro-F1) | label log-likelihood |
| T4 | Extraction to JSON | schema-valid field F1 | greedy generation |
| T5 | Concise answer style | correct ∧ ≤ 4 words | greedy generation |
| T6 | Verbose answer style | correct ∧ 25–80 words | greedy generation |
| T7 | Python code generation | unit-test pass (isolated subprocess) | greedy generation |
| T8 | Arithmetic word problems | exact final number | greedy generation |
| T9 | Lay → clinical terminology | exact match | greedy generation |
| T10 | Format constraint (`- UPPER` lines + `END`) | all constraints satisfied | greedy generation |
| C1 | Arithmetic + concise (Stage 3) | both factors, each also reported separately | greedy generation |

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
