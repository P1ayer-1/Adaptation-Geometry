"""LoRA training with Transformers + PEFT (spec §5.3, §11, §20.2).

- Targets the same *semantic* module classes on every base (explicit omissions recorded).
- Identical token budgets rather than identical epochs.
- Selects the checkpoint with the best *validation* metric; never touches the test split.
- Writes a full run manifest: base revision, dataset hashes, code commit, seed, hardware,
  determinism settings, LoRA config, optimiser config, token counts, selection criterion.
"""

from __future__ import annotations

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
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    determinism = set_determinism(seed, train.deterministic)
    device = pick_device(train.device)
    model, tok, prov = load_base_model(base, dtype=train.dtype, device=str(device))
    fp_before = state_fingerprint(model)
    model, inventory = build_peft_model(model, lora)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    instruction = instruction_for(task)
    template = task.prompt_template_version
    train_rows = load_split(task, "train", data_dir)
    valid_rows = load_split(task, "valid", data_dir)[: train.valid_max_examples]
    enc_train = [encode_example(tok, instruction, ex, template, train.max_seq_len) for ex in train_rows]
    enc_valid = [encode_example(tok, instruction, ex, template, train.max_seq_len) for ex in valid_rows]
    valid_batches = [collate(enc_valid[i:i + train.batch_size], tok.pad_token_id)
                     for i in range(0, len(enc_valid), train.batch_size)]

    avg_len = sum(len(e["input_ids"]) for e in enc_train) / len(enc_train)
    tokens_per_step = avg_len * train.batch_size * train.grad_accum
    est_steps = max(1, math.ceil(train.max_tokens_seen / tokens_per_step))
    warmup = max(1, int(train.warmup_ratio * est_steps))

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
    best: dict[str, Any] = {"value": None, "step": 0}
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

    def is_better(v: float) -> bool:
        if best["value"] is None:
            return True
        if train.selection_metric == "valid_loss":
            return v < best["value"]
        return v > best["value"] if higher else v < best["value"]

    init_eval = evaluate_now()
    log_f.write(json.dumps({"step": 0, "tokens_seen": 0, **init_eval}) + "\n")
    model.train()
    while tokens_seen < train.max_tokens_seen:
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
            break
        if step % train.eval_every_steps == 0 or tokens_seen >= train.max_tokens_seen:
            ev = evaluate_now()
            rec.update(ev)
            if is_better(ev[train.selection_metric]):
                best = {"value": ev[train.selection_metric], "step": step, **ev}
                best_state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
        log_f.write(json.dumps(rec) + "\n")
    log_f.close()
    diverged = not math.isfinite(step_loss)

    set_peft_model_state_dict(model, best_state)
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
        "max_tokens_seen": train.max_tokens_seen,
        "tokens_seen": tokens_seen,
        "steps": step,
        "epochs_started": epoch,
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
