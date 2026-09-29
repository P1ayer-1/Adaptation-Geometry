"""Evaluation with task-specific metric plugins and per-example JSONL outputs (spec §20.4).

Classification tasks (``scoring: choices``) rank the label set by mean per-token
log-likelihood; all other tasks use greedy decoding. Evaluation always reads the immutable
manifest-verified split, and can evaluate the raw base, a PEFT adapter, or a predicted
update (a :class:`~uag.extract_delta.DeltaSet` added to the base weights).
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import BaseConfig, TaskConfig
from .data import instruction_for, load_split
from .metrics import bootstrap_ci, get_metric, macro_f1, word_count
from .prompts import build_fewshot_prompt, stop_marker, target_text


def _device(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def score_choices(model, tok, prompts: Sequence[str], choices: Sequence[Sequence[str]],
                  template: str = "v1", batch_size: int = 16, norm: str = "mean") -> list[list[float]]:
    """Log-likelihood of each choice continuation given its prompt."""
    items = []
    for i, (p, cs) in enumerate(zip(prompts, choices)):
        p_ids = tok(p, add_special_tokens=True)["input_ids"]
        for c in cs:
            c_ids = tok(target_text(c, template), add_special_tokens=False)["input_ids"]
            items.append((i, p_ids, c_ids))
    scores: list[list[float]] = [[] for _ in prompts]
    dev = _device(model)
    pad = tok.pad_token_id
    for b in range(0, len(items), batch_size):
        chunk = items[b:b + batch_size]
        seqs = [p + c for _, p, c in chunk]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s)
            att[j, : len(s)] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits
        for j, (i, p, c) in enumerate(chunk):
            # Only the choice positions go to fp32: a full-sequence fp32 log-softmax over a 128k
            # vocabulary does not fit next to a 1B model on an 8 GB card with few-shot prompts.
            pos = torch.arange(len(p) - 1, len(p) + len(c) - 1, device=logits.device)
            logp = torch.log_softmax(logits[j, pos].float(), -1)
            lp = logp[torch.arange(len(c), device=logits.device), torch.tensor(c, device=logits.device)].sum().item()
            scores[i].append(lp / len(c) if norm == "mean" else lp)
    return scores


@torch.no_grad()
def generate(model, tok, prompts: Sequence[str], max_new_tokens: int, batch_size: int = 16) -> list[str]:
    """Greedy decoding with left padding; returns only the continuation text."""
    dev = _device(model)
    outs: list[str] = []
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        for b in range(0, len(prompts), batch_size):
            enc = tok(list(prompts[b:b + batch_size]), return_tensors="pt", padding=True,
                      add_special_tokens=True).to(dev)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = old_side
    return outs


def select_shots(task: TaskConfig, k: int, seed: int = 0, data_dir: str | Path = "data") -> list[dict[str, Any]]:
    """Deterministic few-shot demonstrations drawn from the *train* split (never valid/test).

    The same k shots are used for every evaluation example of a task. For classification tasks
    the shots cycle through the sorted label set, so every label is shown when k >= #labels.
    The order is then shuffled with the same seeded RNG.
    """
    if k <= 0:
        return []
    rows = load_split(task, "train", data_dir)
    rng = random.Random(f"fewshot:{task.dataset_key}:{seed}")
    if task.scoring == "choices":
        by_label: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_label.setdefault(r["target"], []).append(r)
        pools = {lab: rng.sample(v, len(v)) for lab, v in sorted(by_label.items())}
        labels = sorted(pools)
        shots = [pools[labels[i % len(labels)]][i // len(labels)] for i in range(k)]
        rng.shuffle(shots)
    else:
        shots = rng.sample(rows, k)
    return shots


def truncate_at(text: str, marker: str) -> str:
    i = text.find(marker)
    return text if i < 0 else text[:i]


def evaluate_examples(model, tok, task: TaskConfig, examples: list[dict[str, Any]],
                      template: str = "v1", batch_size: int = 16,
                      shots: Sequence[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score ``examples``; returns (per-example records, summary).

    With ``shots``, every prompt starts with those worked examples (few-shot), and each
    generation is cut where the model starts inventing the next ``Input:`` block.
    """
    instruction = instruction_for(task)
    shots = list(shots or [])
    prompts = [build_fewshot_prompt(instruction, shots, ex["input"], template) for ex in examples]
    metric = get_metric(task.metric)
    t0 = time.time()
    records = []
    if task.scoring == "choices":
        choice_lists = [ex["metadata"]["choices"] for ex in examples]
        scores = score_choices(model, tok, prompts, choice_lists, template, batch_size)
        preds = [cs[int(np.argmax(sc))] for cs, sc in zip(choice_lists, scores)]
        for ex, pred, sc in zip(examples, preds, scores):
            records.append({"example_id": ex["example_id"], "prediction": pred,
                            "score": metric(pred, ex), "choice_logprobs": sc})
    else:
        preds = generate(model, tok, prompts, task.max_new_tokens, batch_size)
        if shots:
            preds = [truncate_at(p, stop_marker(template)) for p in preds]
        for ex, pred in zip(examples, preds):
            records.append({"example_id": ex["example_id"], "prediction": pred, "score": metric(pred, ex)})
    s = [r["score"] for r in records]
    point, lo, hi = bootstrap_ci(s)
    summary: dict[str, Any] = {
        "task_id": task.task_id, "metric": task.metric, "direction": task.metric_direction,
        "primary": point, "ci95": [lo, hi], "n": len(records), "template": template,
        "shots": len(shots), "shot_example_ids": [ex["example_id"] for ex in shots],
        "stop_marker": stop_marker(template) if shots and task.scoring != "choices" else None,
        "mean_output_words": float(np.mean([word_count(r["prediction"]) for r in records])) if records else 0.0,
        "eval_seconds": time.time() - t0,
    }
    if task.factor_metrics:
        ex_by_id = {ex["example_id"]: ex for ex in examples}
        summary["factors"] = {}
        for factor, mname in task.factor_metrics.items():
            fm = get_metric(mname)
            vals = [fm(r["prediction"], ex_by_id[r["example_id"]]) for r in records]
            for r, v in zip(records, vals):
                r.setdefault("factor_scores", {})[factor] = v
            summary["factors"][factor] = float(np.mean(vals)) if vals else float("nan")
    if task.scoring == "choices":
        summary["macro_f1"] = macro_f1([r["prediction"] for r in records],
                                       [ex["target"] for ex in examples])
    return records, summary


def load_eval_examples(task: TaskConfig, split: str, data_dir: str | Path = "data",
                       max_examples: int | None = None) -> list[dict[str, Any]]:
    rows = load_split(task, split, data_dir)
    return rows[:max_examples] if max_examples else rows


def run_evaluation(base: BaseConfig, task: TaskConfig, out_dir: str | Path, eval_id: str, *,
                   adapter_dir: str | Path | None = None, delta_dir: str | Path | None = None,
                   split: str = "test", template: str | None = None, data_dir: str | Path = "data",
                   max_examples: int | None = None, batch_size: int = 16, device: str = "auto",
                   dtype: str | None = None, extra_meta: dict[str, Any] | None = None,
                   model_cache: dict | None = None, fewshot_k: int = 0, fewshot_seed: int = 0) -> dict[str, Any]:
    """Evaluate a base / adapter / predicted update and write ``<eval_id>.jsonl`` + summary.

    ``fewshot_k > 0`` prepends k deterministic training examples to every prompt
    (:func:`select_shots`); the selection is recorded in the summary.
    """
    from .extract_delta import DeltaSet, applied_delta
    from .models import load_base_model

    if adapter_dir and delta_dir:
        raise ValueError("pass either adapter_dir or delta_dir, not both")
    key = (base.name, dtype, device)
    if model_cache is not None and key in model_cache:
        model, tok, prov = model_cache[key]
    else:
        model, tok, prov = load_base_model(base, dtype=dtype, device=device)
        if model_cache is not None:
            model_cache[key] = (model, tok, prov)
    template = template or task.prompt_template_version
    examples = load_eval_examples(task, split, data_dir, max_examples)
    shots = select_shots(task, fewshot_k, fewshot_seed, data_dir)
    ctx: contextlib.AbstractContextManager = contextlib.nullcontext(model)
    kind = "base"
    eval_model = model
    if adapter_dir:
        from peft import PeftModel

        eval_model = PeftModel.from_pretrained(model, str(adapter_dir))
        eval_model.eval()
        kind = "adapter"
    elif delta_dir:
        ctx = applied_delta(model, DeltaSet.load(delta_dir))
        kind = "delta"
    with ctx:
        records, summary = evaluate_examples(eval_model, tok, task, examples, template, batch_size, shots)
    if adapter_dir:
        eval_model.unload()  # restore the cached base model (adapter layers removed)
    summary.update({"eval_id": eval_id, "base": base.name, "family": base.family, "split": split,
                    "kind": kind, "adapter_dir": str(adapter_dir) if adapter_dir else None,
                    "delta_dir": str(delta_dir) if delta_dir else None, "provenance": prov,
                    "fewshot_seed": fewshot_seed if fewshot_k else None,
                    **(extra_meta or {})})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{eval_id}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out / f"{eval_id}.summary.json").write_text(json.dumps(summary, indent=1))
    return summary
