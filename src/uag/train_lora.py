"""LoRA training with Transformers + PEFT (spec §5.3, §11, §20.2).

- Targets the same *semantic* module classes on every base (explicit omissions recorded).
- Identical token budgets rather than identical epochs.
- Selects the checkpoint with the best *validation* metric; never touches the test split.
- Writes a full run manifest: base revision, dataset hashes, code commit, seed, hardware,
  determinism settings, LoRA config, optimiser config, token counts, selection criterion.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .alignment import discover_modules
from .config import BaseConfig, LoraSettings, TaskConfig, TrainSettings, dump_yaml, to_dict
from .data import dataset_hashes, instruction_for, load_split
from .models import load_base_model, pick_device, state_fingerprint
from .prompts import build_prompt, target_text
from .provenance import git_commit, hardware_info, set_determinism, sha256_file


def make_run_id(base: BaseConfig, task: TaskConfig, seed: int, lora: LoraSettings) -> str:
    return f"{base.name}_{task.task_id}_seed{seed}_r{lora.rank}"


def encode_example(tok, instruction: str, ex: dict[str, Any], template: str, max_len: int) -> dict[str, list[int]]:
    """Prompt tokens are masked from the loss; target tokens + EOS are supervised.

    Over-long examples lose prompt tokens from the *left* (the target is never cut unless it
    alone exceeds ``max_len``), so every example keeps a supervised signal.
    """
    p_ids = tok(build_prompt(instruction, ex["input"], template), add_special_tokens=True)["input_ids"]
    t_ids = tok(target_text(ex["target"], template), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    t_ids = t_ids[:max_len]
    keep = max_len - len(t_ids)
    p_ids = p_ids[len(p_ids) - keep:] if keep > 0 else []
    return {"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids}


def collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": att}


def build_peft_model(model, lora: LoraSettings):
    from peft import LoraConfig, get_peft_model

    inv = discover_modules(model, lora.target_modules)
    cfg = LoraConfig(r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                     target_modules=inv.names(), use_rslora=lora.use_rslora, bias="none",
                     task_type="CAUSAL_LM", init_lora_weights=True)
    return get_peft_model(model, cfg), inv


class StaleRunError(RuntimeError):
    pass


def check_reusable(run_dir: Path, task: TaskConfig, lora: LoraSettings, train: TrainSettings,
                   data_dir: str | Path = "data") -> None:
    """A finished run is only reused if it was trained on the same dataset version and LoRA /
    training settings; otherwise the experiment needs a new name (nothing is silently mixed)."""
    import yaml

    m = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    problems = []
    if m.get("dataset_sha256") != dataset_hashes(task, data_dir):
        problems.append(f"dataset {task.dataset_key} differs from the one it was trained on "
                        f"(v{m.get('dataset_version')})")
    if m.get("lora_a_init", "random") != lora.a_init:
        problems.append(f"lora.a_init {lora.a_init} != {m.get('lora_a_init', 'random')}")
    if m.get("lora_train_A", True) != lora.train_A:
        problems.append(f"lora.train_A {lora.train_A} != {m.get('lora_train_A', True)}")
    if m.get("lora_init_seed_scope", "seed") != lora.init_seed_scope:
        problems.append(f"lora.init_seed_scope {lora.init_seed_scope!r} != {m.get('lora_init_seed_scope', 'seed')!r}")
    for k in ("lora_alpha", "lora_dropout", "use_rslora"):
        if m.get(k) != getattr(lora, k.replace("lora_", "") if k != "use_rslora" else k):
            problems.append(f"{k} changed")
    old_train = m.get("train_settings", {})
    changed = [k for k, v in to_dict(train).items() if k in old_train and old_train[k] != v]
    if changed:
        problems.append(f"train settings changed: {changed}")
    if problems:
        raise StaleRunError(f"{run_dir.name} exists but {'; '.join(problems)}. Use a new experiment name "
                            f"(or delete the run deliberately); finished runs are never silently mixed.")


def lora_init_seed(task: TaskConfig, seed: int, lora: LoraSettings, base: BaseConfig | None = None) -> int | None:
    """RNG seed for the random LoRA A init. ``init_seed_scope: seed`` (legacy) leaves the global
    RNG as seeded by the run seed, so every task on a base shares the same A0 for a given seed;
    ``task`` derives a distinct init per (task, seed); ``base`` one init per base."""
    if lora.init_seed_scope == "seed":
        return None
    key = f"lora-init:base:{base.name}" if lora.init_seed_scope == "base" else f"lora-init:{task.task_id}:{seed}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


@torch.no_grad()
def factor_movement(init: dict[str, torch.Tensor], final: dict[str, torch.Tensor],
                    model_type: str | None = None) -> dict[str, Any]:
    """How far each LoRA factor moved from its initialisation.

    - ``a_rel_move`` = ||A - A0|| / ||A0||. Near 0 means A is still essentially its random init.
    - ``b_norm`` = ||B|| (B starts at zero, so this is its whole movement), and ``b_rel_to_a0`` =
      ||B|| / ||A0|| for scale.
    - ``a_rowspace_overlap`` = overlap of the row spaces of A and A0 (1 = identical, chance ~ r/d_in).
      ΔW = scale·BA only acts on inputs in A's row space, so an overlap near 1 means the
      input-side directions of ΔW are fixed by the random init rather than by the task.
    """
    from .spectral import subspace_overlap

    per: dict[str, dict[str, float]] = {}
    for ka, a in final.items():
        if ".lora_A." not in ka:
            continue
        kb = ka.replace(".lora_A.", ".lora_B.")
        a, a0 = a.float().cpu(), init[ka].float().cpu()
        b = final[kb].float().cpu()
        n_a0 = float(a0.norm())
        per[ka.split(".lora_A.")[0]] = {
            "a_rel_move": float((a - a0).norm()) / n_a0 if n_a0 > 0 else float("nan"),
            "b_norm": float(b.norm()),
            "b_rel_to_a0": float(b.norm()) / n_a0 if n_a0 > 0 else float("nan"),
            "a_rowspace_overlap": subspace_overlap(a.double().numpy().T, a0.double().numpy().T),
            "chance_overlap": a.shape[0] / a.shape[1],
        }

    def agg(rows: list[dict[str, float]]) -> dict[str, float]:
        out = {}
        for k in ("a_rel_move", "b_norm", "b_rel_to_a0", "a_rowspace_overlap", "chance_overlap"):
            v = torch.tensor([r[k] for r in rows], dtype=torch.float64)
            out[f"{k}_mean"] = float(v.mean())
            out[f"{k}_median"] = float(v.median())
        out["a_rel_move_min"] = min(r["a_rel_move"] for r in rows)
        out["a_rel_move_max"] = max(r["a_rel_move"] for r in rows)
        return out

    from .alignment import classify_module, strip_peft_prefix

    by_cls: dict[str, list[dict[str, float]]] = {}
    for name, r in per.items():
        cls = classify_module(strip_peft_prefix(name), model_type) if model_type else None
        by_cls.setdefault(cls or "other", []).append(r)
    return {"overall": agg(list(per.values())) if per else {},
            "by_class": {c: agg(v) for c, v in sorted(by_cls.items())}, "modules": per}


@torch.no_grad()
def mean_a_move(model, init: dict[str, torch.Tensor]) -> float:
    """Cheap running version of ``a_rel_move`` (mean over modules) for the train log."""
    from peft import get_peft_model_state_dict

    vals = [float((v.float() - init[k].to(v.device).float()).norm() / init[k].float().norm())
            for k, v in get_peft_model_state_dict(model).items() if ".lora_A." in k]
    return sum(vals) / len(vals) if vals else float("nan")


@torch.no_grad()
def validation_loss(model, batches: list[dict[str, torch.Tensor]], device) -> float:
    model.eval()
    total, count = 0.0, 0
    for b in batches:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()
        shift_logits, shift_labels = logits[:, :-1], b["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                                 shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
        total += loss.item()
        count += int((shift_labels != -100).sum())
    model.train()
    return total / max(count, 1)


def train_lora(base: BaseConfig, task: TaskConfig, lora: LoraSettings, train: TrainSettings, seed: int,
               runs_dir: str | Path, data_dir: str | Path = "data", overwrite: bool = False) -> Path:
    """Train one base × task × seed adapter; returns the run directory."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    run_id = make_run_id(base, task, seed, lora)
    run_dir = Path(runs_dir) / run_id
    if (run_dir / "manifest.yaml").exists() and not overwrite:
        check_reusable(run_dir, task, lora, train, data_dir)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cap = train.token_cap(task.task_id)

    determinism = set_determinism(seed, train.deterministic)
    device = pick_device(train.device)
    model, tok, prov = load_base_model(base, dtype=train.dtype, device=str(device))
    fp_before = state_fingerprint(model)
    if train.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()  # frozen embeddings: let gradients reach LoRA layers
    init_seed = lora_init_seed(task, seed, lora, base)
    if init_seed is not None:
        torch.manual_seed(init_seed)
    model, inventory = build_peft_model(model, lora)
    eyes_info = None
    if lora.a_init != "random":
        from .eyes import apply_eyes

        model.to(device)
        eyes_info = apply_eyes(model, tok, inventory, base.name, lora.a_init, lora.a_calibration,
                               lora.a_calibration_tokens, init_seed, Path(runs_dir).parent / "eyes")
    if init_seed is not None:
        torch.manual_seed(seed)  # the run seed governs everything after the A init (e.g. dropout)
    if not lora.train_A:
        for n, p in model.named_parameters():
            if ".lora_A." in n:
                p.requires_grad_(False)
    model.to(device)
    init_state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    instruction = instruction_for(task)
    template = task.prompt_template_version
    train_rows = load_split(task, "train", data_dir)[: train.max_train_examples or None]
    valid_rows = load_split(task, "valid", data_dir)[: train.valid_max_examples]
    enc_train = [encode_example(tok, instruction, ex, template, train.max_seq_len) for ex in train_rows]
    enc_valid = [encode_example(tok, instruction, ex, template, train.max_seq_len) for ex in valid_rows]
    valid_batches = [collate(enc_valid[i:i + train.batch_size], tok.pad_token_id)
                     for i in range(0, len(enc_valid), train.batch_size)]

    avg_len = sum(len(e["input_ids"]) for e in enc_train) / len(enc_train)
    tokens_per_step = avg_len * train.batch_size * train.grad_accum
    est_steps = max(1, math.ceil(cap / tokens_per_step))
    if train.max_steps is not None:  # schedule (warmup + decay) spans the fixed step budget
        est_steps = min(est_steps, train.max_steps)
    warmup = max(1, int(train.warmup_ratio * est_steps))
    min_steps = max(train.min_steps,
                    math.ceil(train.min_epochs * len(enc_train) / (train.batch_size * train.grad_accum)))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=train.learning_rate,
                            weight_decay=train.weight_decay)
    def lr_lambda(s: int) -> float:  # linear warmup, then linear decay to a 5% floor
        if s < warmup:
            return (s + 1) / warmup
        return max(0.05, 1 - (s - warmup) / max(1, est_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    order_rng = random.Random(seed)
    order: list[int] = []
    tokens_seen, step, epoch = 0, 0, 0
    best: dict[str, Any] = {"value": None, "step": 0, "tokens_seen": 0}
    best_state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    higher = task.metric_direction == "higher"
    log_f = open(run_dir / "train_log.jsonl", "w")
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def evaluate_now() -> dict[str, float]:
        out = {"valid_loss": validation_loss(model, valid_batches, device)}
        if train.selection_metric == "valid_primary":
            from .evaluate import evaluate_examples

            model.eval()
            _, summ = evaluate_examples(model, tok, task, valid_rows, template)
            model.train()
            out["valid_primary"] = summ["primary"]
        return out

    def gain(v: float, ref: float) -> float:  # > 0 means v is better than ref
        lower = train.selection_metric == "valid_loss" or not higher
        return ref - v if lower else v - ref

    def is_better(v: float) -> bool:
        return best["value"] is None or gain(v, best["value"]) > 0

    print(f"[uag {time.strftime('%H:%M:%S')}] training {run_id} (<= {est_steps} steps"
          + (f", early stopping after {train.early_stopping_patience} evals without improvement" if
             train.early_stopping_patience else "") + ")", flush=True)
    init_eval = evaluate_now()
    log_f.write(json.dumps({"step": 0, "tokens_seen": 0, **init_eval}) + "\n")
    model.train()
    stop_reason = "token_cap"
    bad_evals = 0
    while tokens_seen < cap and (train.max_steps is None or step < train.max_steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(train.grad_accum):
            if len(order) < train.batch_size:
                perm = list(range(len(enc_train)))
                order_rng.shuffle(perm)
                order.extend(perm)
                epoch += 1
            idx, order = order[: train.batch_size], order[train.batch_size:]
            batch = collate([enc_train[i] for i in idx], tok.pad_token_id)
            tokens_seen += int(batch["attention_mask"].sum())
            if int((batch["labels"][:, 1:] != -100).sum()) == 0:
                continue  # nothing supervised in this micro-batch
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / train.grad_accum
            loss.backward()
            step_loss += loss.item()
        opt.step()
        sched.step()
        step += 1
        rec: dict[str, Any] = {"step": step, "tokens_seen": tokens_seen, "train_loss": step_loss,
                               "lr": sched.get_last_lr()[0]}
        if not math.isfinite(step_loss):
            rec["diverged"] = True
            log_f.write(json.dumps(rec) + "\n")
            stop_reason = "diverged"
            break
        elapsed = time.time() - t0
        out_of_time = train.max_wall_minutes is not None and elapsed >= 60 * train.max_wall_minutes
        stop = False
        at_max = train.max_steps is not None and step >= train.max_steps
        if step % train.eval_every_steps == 0 or tokens_seen >= cap or out_of_time or at_max:
            ev = evaluate_now()
            rec.update(ev)
            rec["a_rel_move_mean"] = mean_a_move(model, init_state)
            v = ev[train.selection_metric]
            if best["value"] is None or gain(v, best["value"]) > train.early_stopping_min_delta:
                bad_evals = 0
            else:
                bad_evals += 1
            if is_better(v):
                best = {"value": v, "step": step, "tokens_seen": tokens_seen, **ev}
                best_state = {k: w.detach().clone() for k, w in get_peft_model_state_dict(model).items()}
            rec["evals_without_improvement"] = bad_evals
            frac = min(tokens_seen / cap, 1.0)
            print(f"[uag {time.strftime('%H:%M:%S')}]   {run_id}: step {step}, {frac:.0%} of token cap, "
                  f"train loss {step_loss:.3g}, {train.selection_metric} {v:.4f} (best {best['value']:.4f} "
                  f"@ step {best['step']}), <= {elapsed / frac * (1 - frac) / 60:.1f} min left", flush=True)
            if out_of_time:
                stop, stop_reason = True, "wall_clock"
            elif at_max:
                stop, stop_reason = True, "max_steps"
            elif (train.early_stopping_patience and bad_evals >= train.early_stopping_patience
                  and step >= min_steps):
                stop, stop_reason = True, "early_stopping"
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if stop:
            break
    log_f.close()
    diverged = not math.isfinite(step_loss)

    set_peft_model_state_dict(model, best_state)
    movement = factor_movement(init_state, best_state, inventory.model_type)
    (run_dir / "factor_movement.json").write_text(json.dumps(movement, indent=1))
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    fp_after = state_fingerprint(model)

    manifest = {
        "run_id": run_id,
        "base_model": base.model_id,
        "base_revision": base.revision,
        "base_name": base.name,
        "base_family": base.family,
        "base_role": base.role,
        "base_provenance": prov,
        "base_frozen_fingerprint": {"before": fp_before, "after": fp_after, "unchanged": fp_before == fp_after},
        "task_id": task.task_id,
        "dataset_version": task.version,
        "dataset_sha256": dataset_hashes(task, data_dir),
        "prompt_template_version": template,
        "seed": seed,
        "lora_rank": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "use_rslora": lora.use_rslora,
        "lora_scaling": lora.scaling,
        "lora_init_seed_scope": lora.init_seed_scope,
        "lora_train_A": lora.train_A,
        "lora_a_init": lora.a_init,
        "lora_eyes": eyes_info,
        "lora_init_seed": init_seed if init_seed is not None else seed,
        "factor_movement": {"overall": movement["overall"], "by_class": movement["by_class"],
                            "detail": "factor_movement.json"},
        "target_modules": [c for c in lora.target_modules if c not in inventory.omissions],
        "target_module_names": inventory.names(),
        "module_omissions": inventory.omissions,
        "module_inventory": inventory.to_dict(),
        "dtype": train.dtype,
        "optimizer": train.optimizer,
        "learning_rate": train.learning_rate,
        "weight_decay": train.weight_decay,
        "batch_size": train.batch_size,
        "grad_accum": train.grad_accum,
        "max_tokens_seen": cap,
        "token_cap_source": "max_tokens_by_task" if task.task_id in train.max_tokens_by_task else "max_tokens_seen",
        "tokens_seen": tokens_seen,
        "n_train_examples": len(train_rows),
        "steps": step,
        "epochs_started": epoch,
        "stopping": {"reason": stop_reason, "step": step, "tokens_seen": tokens_seen,
                     "best_step": best["step"], "best_tokens_seen": best.get("tokens_seen"),
                     "patience_evals": train.early_stopping_patience, "min_delta": train.early_stopping_min_delta,
                     "min_steps": min_steps, "min_epochs": train.min_epochs, "max_tokens_seen": cap,
                     "eval_every_steps": train.eval_every_steps},
        "warmup_steps": warmup,
        "selection": {"metric": train.selection_metric, "split": "valid",
                      "n_valid": len(valid_rows), **best, "initial": init_eval},
        "diverged": diverged,
        "trainable_params": trainable,
        "total_params": total_params,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "wall_seconds": time.time() - t0,
        "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "code_commit": git_commit(),
        "uag_version": __version__,
        "hardware": hardware_info(),
        "determinism": determinism,
        "train_settings": to_dict(train),
    }
    dump_yaml(manifest, run_dir / "manifest.yaml")
    return run_dir


def load_trained(base: BaseConfig, run_dir: str | Path, device: str = "auto", dtype: str | None = None):
    """Reload an adapter into the exact frozen base. Returns (peft_model, tokenizer)."""
    import yaml
    from peft import PeftModel

    manifest = yaml.safe_load((Path(run_dir) / "manifest.yaml").read_text())
    model, tok, prov = load_base_model(base, dtype=dtype or manifest["dtype"], device=device)
    if base.source == "hub" and manifest["base_revision"] != base.revision:
        raise RuntimeError("base revision differs from the one the adapter was trained on")
    if "weights_sha256" in prov and prov["weights_sha256"] != manifest["base_provenance"].get("weights_sha256"):
        raise RuntimeError("base weights differ from the ones the adapter was trained on")
    peft_model = PeftModel.from_pretrained(model, str(Path(run_dir) / "adapter"))
    peft_model.eval()
    return peft_model, tok
