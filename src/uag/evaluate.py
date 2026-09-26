"""Evaluation with task-specific metric plugins and per-example JSONL outputs (spec §20.4).

Classification tasks (``scoring: choices``) rank the label set by mean per-token
log-likelihood; all other tasks use greedy decoding. Evaluation always reads the immutable
manifest-verified split, and can evaluate the raw base, a PEFT adapter, or a predicted
update (a :class:`~uag.extract_delta.DeltaSet` added to the base weights).
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import BaseConfig, TaskConfig
from .data import instruction_for, load_split
from .metrics import bootstrap_ci, get_metric, macro_f1, word_count
from .prompts import build_prompt, target_text


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
        logp = torch.log_softmax(model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float(), -1)
        for j, (i, p, c) in enumerate(chunk):
            pos = torch.arange(len(p) - 1, len(p) + len(c) - 1)
            lp = logp[j, pos, torch.tensor(c)].sum().item()
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


def evaluate_examples(model, tok, task: TaskConfig, examples: list[dict[str, Any]],
                      template: str = "v1", batch_size: int = 16) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score ``examples``; returns (per-example records, summary)."""
    instruction = instruction_for(task)
    prompts = [build_prompt(instruction, ex["input"], template) for ex in examples]
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
        for ex, pred in zip(examples, preds):
            records.append({"example_id": ex["example_id"], "prediction": pred, "score": metric(pred, ex)})
    s = [r["score"] for r in records]
    point, lo, hi = bootstrap_ci(s)
    summary: dict[str, Any] = {
        "task_id": task.task_id, "metric": task.metric, "direction": task.metric_direction,
        "primary": point, "ci95": [lo, hi], "n": len(records), "template": template,
        "mean_output_words": float(np.mean([word_count(r["prediction"]) for r in records])) if records else 0.0,
        "eval_seconds": time.time() - t0,
    }
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
                   model_cache: dict | None = None) -> dict[str, Any]:
    """Evaluate a base / adapter / predicted update and write ``<eval_id>.jsonl`` + summary."""
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
        records, summary = evaluate_examples(eval_model, tok, task, examples, template, batch_size)
    if adapter_dir:
        eval_model.unload()  # restore the cached base model (adapter layers removed)
    summary.update({"eval_id": eval_id, "base": base.name, "family": base.family, "split": split,
                    "kind": kind, "adapter_dir": str(adapter_dir) if adapter_dir else None,
                    "delta_dir": str(delta_dir) if delta_dir else None, "provenance": prov,
                    **(extra_meta or {})})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{eval_id}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out / f"{eval_id}.summary.json").write_text(json.dumps(summary, indent=1))
    return summary
