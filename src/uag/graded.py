"""Graded transfer: how much more likely does a predicted update make the *gold* answers?

The task metrics of the held-out tasks are all-or-nothing per example (an exact JSON record,
an exact final number), so an update that moves the model part of the way still scores 0.
This diagnostic measures the mean per-token negative log-likelihood (NLL) of the gold target
on test examples (prompt tokens masked, as in training) for the raw base, the direct adapter
(its exact ΔW), and every predicted update, and reports

    RecoveredNLL = (NLL_base - NLL_predicted) / (NLL_base - NLL_direct)

which is 0 for no help and 1 for as good as direct LoRA (negative = harmful). It reads only
existing predictions and runs; it is a diagnostic next to the predeclared gate, not part of it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .data import instruction_for, load_split
from .extract_delta import DeltaSet, applied_delta
from .pipeline import Paths, _filter


@torch.no_grad()
def gold_nll(model, tok, task, examples: list[dict[str, Any]], batch_size: int = 8, max_len: int = 512) -> float:
    """Mean per-token NLL of the gold targets (token-weighted over all examples)."""
    from .train_lora import collate, encode_example

    dev = next(model.parameters()).device
    total, count = 0.0, 0
    enc = [encode_example(tok, instruction_for(task), ex, task.prompt_template_version, max_len) for ex in examples]
    for i in range(0, len(enc), batch_size):
        b = {k: v.to(dev) for k, v in collate(enc[i:i + batch_size], tok.pad_token_id).items()}
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        labels = b["labels"][:, 1:]
        mask = labels != -100
        logp = torch.log_softmax(logits[:, :-1][mask].float(), -1)
        total += float(-logp[torch.arange(logp.shape[0], device=dev), labels[mask]].sum())
        count += int(mask.sum())
    return total / max(count, 1)


def graded_transfer(exp: ExperimentConfig, n_examples: int = 100, only_bases: list[str] | None = None) -> dict[str, Any]:
    from .models import load_base_model

    paths = Paths(exp)
    preds: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(paths.preds.glob("*/*/delta_meta.json")) if paths.preds.exists() else []:
        preds[json.loads(d.read_text())["meta"]["target_base"]].append(d.parent)
    rows = []
    for base in _filter(exp.bases, only_bases, lambda b: b.name):
        if base.name not in preds:
            continue
        model, tok, _ = load_base_model(base, dtype=exp.train.dtype, device=exp.train.device)
        model.eval()
        ref: dict[tuple[str, int], tuple[float, float]] = {}
        for d in preds[base.name]:
            meta = json.loads((d / "delta_meta.json").read_text())["meta"]
            task, seed = exp.task(meta["task_id"]), meta["seed"]
            examples = load_split(task, exp.eval_split, exp.data_dir)[:n_examples]
            if (task.task_id, seed) not in ref:
                nll_base = gold_nll(model, tok, task, examples)
                run = paths.run_dir(base.name, task.task_id, seed) / "spectral"
                with applied_delta(model, DeltaSet.load(run)):
                    nll_direct = gold_nll(model, tok, task, examples)
                ref[(task.task_id, seed)] = (nll_base, nll_direct)
            nll_base, nll_direct = ref[(task.task_id, seed)]
            with applied_delta(model, DeltaSet.load(d)):
                nll_pred = gold_nll(model, tok, task, examples)
            gain = nll_base - nll_direct
            rows.append({"target": base.name, "source": meta["source_base"], "task": task.task_id, "seed": seed,
                         "method": meta["method"], "nll_base": nll_base, "nll_direct": nll_direct,
                         "nll_pred": nll_pred,
                         "recovered_nll": (nll_base - nll_pred) / gain if gain > 0 else None})
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    out = {"n_examples": n_examples, "split": exp.eval_split, "rows": rows}
    f = paths.exp_results / "graded_transfer.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    out["saved_to"] = str(f)
    return out


def render_graded(res: dict[str, Any]) -> list[str]:
    acc: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in res["rows"]:
        acc[(r["source"], r["target"], r["task"], r["method"])].append(r)
    L = [f"Graded transfer on {res['n_examples']} {res['split']} examples: mean gold-answer NLL per token "
         f"(base -> direct) and RecoveredNLL = (base - predicted) / (base - direct), seed-averaged",
         f"  {'source -> target':<38}{'task':<15}{'method':<16}{'base':>7}{'direct':>8}{'pred':>8}{'RecNLL':>8}"]
    for (s, t, task, m), rs in sorted(acc.items()):
        rec = [r["recovered_nll"] for r in rs if r["recovered_nll"] is not None]
        L.append(f"  {s + ' -> ' + t:<38}{task:<15}{m:<16}{np.mean([r['nll_base'] for r in rs]):>7.3f}"
                 f"{np.mean([r['nll_direct'] for r in rs]):>8.3f}{np.mean([r['nll_pred'] for r in rs]):>8.3f}"
                 f"{(np.mean(rec) if rec else float('nan')):>8.3f}")
    return L
