"""Shared adapter with per-model connectors ("an adapter for the adapter").

Every base m gets, for each adapted linear module (layer l, class c), a connector pair:

    P_{m,l,c}: d_shared x d_in   (read the module input into a shared code)
    Q_{m,l,c}: d_out x d_shared  (write a shared-code output back into the model)

and every task t is a set of shared cores C_{t,s,c} (d_shared x d_shared), one per layer
*slot* s (layers matched by relative depth) and class. The update is

    ΔW_{m,t,(l,c)} = scale · Q_{m,l,c} · C_{t,slot(l),c} · P_{m,l,c}

which is the spec's Δ(m,t) = D_m(z_t) with z_t = the cores and D_m = the connectors.

Protocol (held-out *tasks*; cf. PorTAL, which ports known tasks to held-out *models*):

1. ``connectors``: train P, Q of all bases and the cores of the map-training tasks jointly,
   every step applying the same task core in every base (this forces a shared code).
   Held-out tasks never enter this phase (:class:`HoldoutLeakError`).
2. ``heldout``: freeze the connectors; learn a held-out task's core on ONE base only.
   A core trained on the target itself through its connectors is the in-system ceiling.
3. ``evaluate``: plug the core into each other base (no target training) and score it with
   the task metric and gold-answer NLL, against an untrained (identity) core, the mean training
   core, a norm-matched random core and the target-trained ceiling.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .alignment import discover_modules
from .config import ExperimentConfig, read_config_file
from .data import instruction_for, load_split
from .transfer import HoldoutLeakError


@dataclass
class SharedSettings:
    d_shared: int = 64
    scale: float = 2.0
    connector_steps: int = 3000
    lr_connectors: float = 2e-4
    lr_cores: float = 2e-4
    lr_core_heldout: float = 1e-3
    eval_every: int = 100
    graded_examples: int = 100
    seed: int = 0
    grad_clip: float | None = None  # max gradient norm (phase 1 and 2); width 128 diverged without it


def load_shared_settings(path: str | Path) -> SharedSettings:
    data = read_config_file(path).get("shared", {}) or {}
    return SharedSettings(**data)


def log(msg: str) -> None:
    print(f"[uag {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# The adapted system: frozen bases + hooks
# ---------------------------------------------------------------------------


class SharedSystem:
    def __init__(self, exp: ExperimentConfig, settings: SharedSettings, device: str = "auto"):
        from .models import load_base_model, pick_device

        self.exp, self.s = exp, settings
        self.device = pick_device(device if device != "auto" else exp.train.device)
        self.models, self.toks, self.mods = {}, {}, {}
        for b in exp.bases:
            model, tok, _ = load_base_model(b, dtype=exp.train.dtype, device=str(self.device))
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
            self.models[b.name], self.toks[b.name] = model, tok
            self.mods[b.name] = discover_modules(model, exp.lora.target_modules)
        self.n_slots = min(inv.num_layers for inv in self.mods.values())
        g = torch.Generator().manual_seed(settings.seed)
        d = settings.d_shared
        self.P, self.Q = {}, {}
        for b, inv in self.mods.items():
            for m in inv.modules:
                key = f"{b}|{m.name}"
                self.P[key] = (torch.randn(d, m.in_features, generator=g) / math.sqrt(m.in_features)).to(self.device)
                self.Q[key] = torch.zeros(m.out_features, d, device=self.device)
        self.classes = sorted({m.cls for inv in self.mods.values() for m in inv.modules})
        self.state: dict[str, Any] = {"cores": None, "base": None, "on": False}
        self.handles = []
        for b, inv in self.mods.items():
            for m in inv.modules:
                mod = self.models[b].get_submodule(m.name)
                self.handles.append(mod.register_forward_hook(self._hook(b, m)))

    def slot(self, base: str, layer: int) -> int:
        return min(int(layer * self.n_slots / self.mods[base].num_layers), self.n_slots - 1)

    def _hook(self, base: str, m):
        key = f"{base}|{m.name}"

        def hook(_mod, inp, out):
            st = self.state
            if not st["on"] or st["base"] != base:
                return out
            c = st["cores"][f"{self.slot(base, m.layer)}|{m.cls}"]
            x = inp[0].float()
            delta = ((x @ self.P[key].T) @ c.T) @ self.Q[key].T
            return out + (self.s.scale * delta).to(out.dtype)

        return hook

    def new_cores(self, init: str = "identity", like: dict[str, torch.Tensor] | None = None,
                  gen: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        d = self.s.d_shared
        keys = [f"{s}|{c}" for s in range(self.n_slots) for c in self.classes]
        if init == "identity":
            return {k: torch.eye(d, device=self.device) for k in keys}
        if init == "random":  # norm-matched to ``like`` core by core
            out = {}
            for k in keys:
                r = torch.randn(d, d, generator=gen).to(self.device)
                out[k] = r * (like[k].norm() / r.norm())
            return out
        raise ValueError(init)

    @contextlib.contextmanager
    def active(self, base: str, cores: dict[str, torch.Tensor]):
        old = dict(self.state)
        self.state.update(cores=cores, base=base, on=True)
        try:
            yield self.models[base]
        finally:
            self.state.update(old)

    def delta_set(self, base: str, cores: dict[str, torch.Tensor]):
        """The exact per-module ΔW = scale·Q C P as a DeltaSet (for reuse with applied_delta)."""
        from .extract_delta import DeltaSet, from_factors

        mods, info = {}, {}
        for m in self.mods[base].modules:
            key = f"{base}|{m.name}"
            c = cores[f"{self.slot(base, m.layer)}|{m.cls}"]
            left = (self.Q[key] @ c).double().cpu().numpy()
            mods[m.name] = from_factors(left, self.P[key].double().cpu().numpy(), self.s.scale)
            info[m.name] = {"layer": m.layer, "cls": m.cls}
        return DeltaSet({"kind": "shared_adapter", "base": base}, mods, info)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _encode(system: SharedSystem, base: str, task, split: str, limit: int | None = None):
    from .train_lora import encode_example

    rows = load_split(task, split, system.exp.data_dir)
    rows = rows[:limit] if limit else rows
    tok = system.toks[base]
    return [encode_example(tok, instruction_for(task), ex, task.prompt_template_version,
                           system.exp.train.max_seq_len) for ex in rows]


def _loss(system: SharedSystem, base: str, batch: list[dict[str, list[int]]]) -> torch.Tensor:
    from .train_lora import collate

    b = {k: v.to(system.device) for k, v in collate(batch, system.toks[base].pad_token_id).items()}
    return system.models[base](**b).loss


@torch.no_grad()
def _valid_loss(system, base, enc, cores, bs: int) -> float:
    with system.active(base, cores):
        losses = [float(_loss(system, base, enc[i:i + bs])) for i in range(0, len(enc), bs)]
    return sum(losses) / max(len(losses), 1)


# ---------------------------------------------------------------------------
# Phase 1: connectors (training tasks only)
# ---------------------------------------------------------------------------


def train_connectors(system: SharedSystem, train_tasks: list[str], holdout: list[str], out_dir: Path) -> dict[str, Any]:
    if set(train_tasks) & set(holdout):
        raise HoldoutLeakError(f"held-out tasks {sorted(set(train_tasks) & set(holdout))} in connector training")
    exp, s, tr = system.exp, system.s, system.exp.train
    rng = random.Random(s.seed)
    tasks = [exp.task(t) for t in train_tasks]
    enc = {(b, t.task_id): _encode(system, b, t, "train") for b in system.models for t in tasks}
    val = {(b, t.task_id): _encode(system, b, t, "valid", tr.valid_max_examples) for b in system.models for t in tasks}
    cores = {t.task_id: system.new_cores("identity") for t in tasks}
    conn = list(system.P.values()) + list(system.Q.values())
    for p in conn:
        p.requires_grad_(True)
    core_params = [c for cs in cores.values() for c in cs.values()]
    for c in core_params:
        c.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": conn, "lr": s.lr_connectors}, {"params": core_params, "lr": s.lr_cores}],
                            weight_decay=0.0)
    warm = max(1, int(0.03 * s.connector_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else max(0.05, 1 - (i - warm) / max(1, s.connector_steps - warm)))
    best, history = None, []
    log_f = open(out_dir / "connector_log.jsonl", "w")
    t0 = time.time()
    for step in range(1, s.connector_steps + 1):
        t = tasks[(step - 1) % len(tasks)].task_id
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for b in system.models:  # the same task core in every base: forces a shared code
            for _ in range(tr.grad_accum):
                batch = [enc[(b, t)][rng.randrange(len(enc[(b, t)]))] for _ in range(tr.batch_size)]
                with system.active(b, cores[t]):
                    loss = _loss(system, b, batch) / (tr.grad_accum * len(system.models))
                loss.backward()
                total += float(loss.detach())
        if s.grad_clip:
            torch.nn.utils.clip_grad_norm_(conn + core_params, s.grad_clip)
        opt.step()
        sched.step()
        rec = {"step": step, "task": t, "train_loss": total}
        if step % s.eval_every == 0 or step == s.connector_steps:
            vl = {f"{b}|{tt.task_id}": _valid_loss(system, b, val[(b, tt.task_id)], cores[tt.task_id], tr.batch_size)
                  for b in system.models for tt in tasks}
            mean = sum(vl.values()) / len(vl)
            rec.update(valid=vl, valid_mean=mean)
            if best is None or mean < best["valid_mean"]:
                best = {"step": step, "valid_mean": mean, "valid": vl,
                        "P": {k: v.detach().clone() for k, v in system.P.items()},
                        "Q": {k: v.detach().clone() for k, v in system.Q.items()},
                        "cores": {tt: {k: c.detach().clone() for k, c in cs.items()} for tt, cs in cores.items()}}
            log(f"connectors step {step}/{s.connector_steps}: train {total:.3f}, mean valid loss {mean:.4f} "
                f"(best {best['valid_mean']:.4f} @ {best['step']}), {(time.time() - t0) / 60:.1f} min")
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        history.append(rec)
    log_f.close()
    with torch.no_grad():
        for k in system.P:
            system.P[k].copy_(best["P"][k])
            system.Q[k].copy_(best["Q"][k])
    for p in conn:
        p.requires_grad_(False)
    save_state(system, best["cores"], out_dir)
    meta = {"train_tasks": train_tasks, "holdout": holdout, "best_step": best["step"],
            "best_valid_mean": best["valid_mean"], "best_valid": best["valid"], "settings": asdict(s),
            "n_slots": system.n_slots, "bases": list(system.models), "minutes": (time.time() - t0) / 60}
    (out_dir / "connectors_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def save_state(system: SharedSystem, train_cores: dict[str, dict[str, torch.Tensor]], out_dir: Path) -> None:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({f"P::{k}": v.contiguous().cpu() for k, v in system.P.items()}
              | {f"Q::{k}": v.contiguous().cpu() for k, v in system.Q.items()}, str(out_dir / "connectors.safetensors"))
    save_file({f"{t}::{k}": c.contiguous().cpu() for t, cs in train_cores.items() for k, c in cs.items()},
              str(out_dir / "train_cores.safetensors"))


def load_state(system: SharedSystem, out_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    from safetensors.torch import load_file

    w = load_file(str(out_dir / "connectors.safetensors"))
    with torch.no_grad():
        for k in system.P:
            system.P[k].copy_(w[f"P::{k}"].to(system.device))
            system.Q[k].copy_(w[f"Q::{k}"].to(system.device))
    cores: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in load_file(str(out_dir / "train_cores.safetensors")).items():
        t, ck = k.split("::", 1)
        cores.setdefault(t, {})[ck] = v.to(system.device)
    return cores


# ---------------------------------------------------------------------------
# Phase 2: a held-out task's core, learned on one base
# ---------------------------------------------------------------------------


def train_core(system: SharedSystem, task_id: str, base: str, out_path: Path) -> dict[str, Any]:
    from .evaluate import evaluate_examples

    exp, s, tr = system.exp, system.s, system.exp.train
    task = exp.task(task_id)
    rng = random.Random(s.seed + 1)
    enc = _encode(system, base, task, "train")
    valid_rows = load_split(task, "valid", exp.data_dir)[: tr.valid_max_examples]
    cores = system.new_cores("identity")
    params = list(cores.values())
    for c in params:
        c.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=s.lr_core_heldout, weight_decay=0.0)
    tokens_per_step = sum(len(e["input_ids"]) for e in enc[:200]) / min(len(enc), 200) * tr.batch_size * tr.grad_accum
    max_steps = max(1, int(tr.token_cap(task_id) / tokens_per_step))
    min_steps = max(tr.min_steps, math.ceil(tr.min_epochs * len(enc) / (tr.batch_size * tr.grad_accum)))

    def score() -> float:
        with torch.no_grad(), system.active(base, cores) as model:
            return evaluate_examples(model, system.toks[base], task, valid_rows)[1]["primary"]

    best = {"value": score(), "step": 0, "cores": {k: c.detach().clone() for k, c in cores.items()}}
    bad, step = 0, 0
    for step in range(1, max_steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(tr.grad_accum):
            batch = [enc[rng.randrange(len(enc))] for _ in range(tr.batch_size)]
            with system.active(base, cores):
                (_loss(system, base, batch) / tr.grad_accum).backward()
        if s.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, s.grad_clip)
        opt.step()
        if step % tr.eval_every_steps == 0:
            v = score()
            if v > best["value"] + tr.early_stopping_min_delta:
                bad = 0
            else:
                bad += 1
            if v > best["value"]:
                best = {"value": v, "step": step, "cores": {k: c.detach().clone() for k, c in cores.items()}}
            log(f"core {task_id} on {base}: step {step}, valid {v:.4f} (best {best['value']:.4f} @ {best['step']})")
            if tr.early_stopping_patience and bad >= tr.early_stopping_patience and step >= min_steps:
                break
    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: c.contiguous().cpu() for k, c in best["cores"].items()}, str(out_path))
    meta = {"task": task_id, "trained_on": base, "best_valid": best["value"], "best_step": best["step"],
            "steps": step, "min_steps": min_steps}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta


def load_cores(system: SharedSystem, path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return {k: v.to(system.device) for k, v in load_file(str(path)).items()}


# ---------------------------------------------------------------------------
# Phase 3: transfer evaluation
# ---------------------------------------------------------------------------


def evaluate_transfer(system: SharedSystem, holdout: list[str], out_dir: Path) -> dict[str, Any]:
    from .evaluate import evaluate_examples
    from .graded import gold_nll

    exp, s = system.exp, system.s
    train_cores = load_state(system, out_dir)
    mean_core = {k: sum(cs[k] for cs in train_cores.values()) / len(train_cores) for k in next(iter(train_cores.values()))}
    gen = torch.Generator().manual_seed(s.seed + 2)
    rows = []
    for task_id in holdout:
        task = exp.task(task_id)
        test = load_split(task, exp.eval_split, exp.data_dir)[: exp.eval_max_examples]
        graded = test[: s.graded_examples]
        own = {b: load_cores(system, out_dir / "heldout" / f"{task_id}__on_{b}.safetensors") for b in system.models}
        for tgt in system.models:
            tok = system.toks[tgt]
            with torch.no_grad():
                base_score = evaluate_examples(system.models[tgt], tok, task, test)[1]["primary"]
                base_nll = gold_nll(system.models[tgt], tok, task, graded)
            candidates = {"ceiling (core trained on target)": own[tgt], "identity core": system.new_cores("identity"),
                          "mean training core": mean_core}
            for src in system.models:
                if src != tgt:
                    candidates[f"transferred from {src}"] = own[src]
                    candidates[f"random (norm of {src} core)"] = system.new_cores("random", like=own[src], gen=gen)
            ceiling_nll = None
            for name, cores in candidates.items():
                with torch.no_grad(), system.active(tgt, cores) as model:
                    score = evaluate_examples(model, tok, task, test)[1]["primary"]
                    nll = gold_nll(model, tok, task, graded)
                if name.startswith("ceiling"):
                    ceiling_nll = nll
                rows.append({"task": task_id, "target": tgt, "condition": name, "score": score, "nll": nll,
                             "base_score": base_score, "base_nll": base_nll})
                log(f"eval {task_id} on {tgt} [{name}]: {task.metric}={score:.3f}, gold NLL {nll:.3f} (base {base_nll:.3f})")
            for r in rows:
                if r["task"] == task_id and r["target"] == tgt:
                    gain = base_nll - ceiling_nll
                    r["recovered_nll_vs_ceiling"] = (base_nll - r["nll"]) / gain if gain > 0 else None
    res = {"rows": rows, "holdout": holdout}
    (out_dir / "transfer_results.json").write_text(json.dumps(res, indent=1))
    return res


def render(res: dict[str, Any]) -> str:
    L = [f"{'task':<15}{'target':<18}{'condition':<40}{'score':>7}{'NLL':>8}{'RecNLL':>8}"]
    for r in res["rows"]:
        rec = r.get("recovered_nll_vs_ceiling")
        L.append(f"{r['task']:<15}{r['target']:<18}{r['condition']:<40}{r['score']:>7.3f}{r['nll']:>8.3f}"
                 f"{(f'{rec:.3f}' if rec is not None else '–'):>8}")
    L.append("RecNLL = (base NLL - NLL) / (base NLL - ceiling NLL); base = the untouched target model.")
    return "\n".join(L)


def run(exp: ExperimentConfig, settings: SharedSettings, phase: str = "all") -> str:
    if not exp.splits or len(exp.splits) != 1:
        raise ValueError("the shared-adapter experiment needs exactly one split (its held-out tasks)")
    holdout = list(exp.splits[0].holdout)
    train_tasks = [t.task_id for t in exp.tasks if t.task_id not in holdout]
    out_dir = exp.path("artifacts") / exp.name / "shared"
    out_dir.mkdir(parents=True, exist_ok=True)
    system = SharedSystem(exp, settings)
    if phase in ("all", "connectors") and not (out_dir / "connectors_meta.json").exists():
        train_connectors(system, train_tasks, holdout, out_dir)
    if phase in ("all", "heldout", "evaluate"):
        load_state(system, out_dir)
    if phase in ("all", "heldout"):
        for t in holdout:
            for b in system.models:
                p = out_dir / "heldout" / f"{t}__on_{b}.safetensors"
                if not p.exists():
                    train_core(system, t, b, p)
    if phase in ("all", "evaluate"):
        res = evaluate_transfer(system, holdout, out_dir)
        text = render(res)
        res_dir = exp.path("results") / exp.name
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "shared_adapter.txt").write_text(text + "\n")
        (res_dir / "shared_adapter.json").write_text(json.dumps(res, indent=1))
        return text
    return f"phase {phase} done ({out_dir})"
