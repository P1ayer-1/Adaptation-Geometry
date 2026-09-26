"""Config-driven Stage-0 orchestration: data → train grid → evals → maps → transfer evals.

Every stage is idempotent (existing outputs are reused) and launchable from an experiment
config with no code edits (spec §13). Failures and exclusions are logged machine-readably
to ``results/<experiment>/exclusions.jsonl``; nothing is silently rerun (spec §21).
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from safetensors.numpy import load_file, save_file

from .config import ExperimentConfig, to_dict
from .data import generate_dataset, instruction_for, load_split
from .extract_delta import DeltaSet, extract_deltas, verify_delta_reconstruction
from .provenance import git_commit, sha256_bytes
from .spectral import save_spectral
from .transfer import (PairMap, TaskSplit, base_weight_topk, cross_lora_params, fit_pair_map,
                       reconstruction_report)


def log(msg: str) -> None:
    print(f"[uag {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Paths:
    def __init__(self, exp: ExperimentConfig):
        self.exp = exp
        art, res = exp.path("artifacts"), exp.path("results")
        self.runs = art / exp.name / "runs"
        self.maps = art / exp.name / "maps"
        self.preds = art / exp.name / "predictions"
        self.base_svd = art / exp.name / "base_svd"
        self.raw = res / "raw" / exp.name
        self.exp_results = res / exp.name
        self.tables = res / "tables"
        self.decision = res / f"{exp.name}_decision.md"
        self.exclusions = self.exp_results / "exclusions.jsonl"
        self.gate_declaration = self.exp_results / "gate_declaration.json"

    def run_dir(self, base: str, task: str, seed: int) -> Path:
        return self.runs / f"{base}_{task}_seed{seed}_r{self.exp.lora.rank}"


def record_exclusion(paths: Paths, **entry: Any) -> None:
    paths.exclusions.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    with open(paths.exclusions, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_exclusions(paths: Paths) -> list[dict[str, Any]]:
    if not paths.exclusions.exists():
        return []
    return [json.loads(l) for l in paths.exclusions.read_text().splitlines() if l.strip()]


def _filter(items: Iterable, only: list[str] | None, key=lambda x: x):
    return [x for x in items if not only or key(x) in only]


# ---------------------------------------------------------------------------
# Stage: data
# ---------------------------------------------------------------------------


def prepare_data(exp: ExperimentConfig) -> None:
    for task in exp.tasks:
        m = generate_dataset(task, exp.data_dir)
        log(f"data {task.dataset_key}: test sha256 {m['test_sha256'][:12]}")


# ---------------------------------------------------------------------------
# Stage: train grid (+ ΔW extraction, verification, spectral cache)
# ---------------------------------------------------------------------------


def _verification_batch(tok, task, n: int = 4):
    from .train_lora import collate, encode_example

    rows = load_split(task, "valid")[:n]
    b = collate([encode_example(tok, instruction_for(task), e, task.prompt_template_version, 256) for e in rows],
                tok.pad_token_id)
    b.pop("labels")
    return b


def postprocess_run(exp: ExperimentConfig, run_dir: Path, base, task) -> dict[str, Any]:
    """Extract ΔW, verify it against the PEFT forward pass, cache factored spectra."""
    from .train_lora import load_trained

    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    spec_dir = run_dir / "spectral"
    if (spec_dir / "spectral_summary.json").exists() and (run_dir / "delta_verification.json").exists():
        return json.loads((run_dir / "delta_verification.json").read_text())
    inv = manifest["module_inventory"]
    meta = {"run_id": manifest["run_id"], "base_name": base.name, "family": base.family, "task_id": task.task_id,
            "seed": manifest["seed"], "num_layers": inv["num_layers"], "module_omissions": inv["omissions"]}
    pm, tok = load_trained(base, run_dir, device=exp.train.device)
    ds = extract_deltas(run_dir / "adapter", manifest["module_inventory"]["model_type"], meta)
    batch = {k: v.to(next(pm.parameters()).device) for k, v in _verification_batch(tok, task).items()}
    atol = 1e-4 if manifest["dtype"] == "float32" else 5e-2
    report = verify_delta_reconstruction(pm, ds, batch, atol=atol)
    (run_dir / "delta_verification.json").write_text(json.dumps(report, indent=1))
    save_spectral(ds, spec_dir)
    return report


def train_grid(exp: ExperimentConfig, only_bases: list[str] | None = None, only_tasks: list[str] | None = None,
               only_seeds: list[int] | None = None) -> None:
    from .train_lora import train_lora

    paths = Paths(exp)
    for base in _filter(exp.bases, only_bases, lambda b: b.name):
        for task in _filter(exp.tasks, only_tasks, lambda t: t.task_id):
            for seed in _filter(exp.seeds, only_seeds):
                t0 = time.time()
                run_dir = train_lora(base, task, exp.lora, exp.train, seed, paths.runs, exp.data_dir)
                manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
                if manifest.get("diverged"):
                    record_exclusion(paths, kind="run", reason="training_divergence", run_id=manifest["run_id"])
                    log(f"train {run_dir.name}: DIVERGED (excluded)")
                    continue
                rep = postprocess_run(exp, run_dir, base, task)
                if not rep["passed"]:
                    record_exclusion(paths, kind="run", reason="delta_verification_failed",
                                     run_id=manifest["run_id"], report=rep)
                log(f"train {run_dir.name}: best {manifest['selection']['metric']}="
                    f"{manifest['selection']['value']:.4f} steps={manifest['steps']} "
                    f"Δ-check={'ok' if rep['passed'] else 'FAIL'} ({time.time() - t0:.0f}s)")


def load_run_deltas(exp: ExperimentConfig, base: str, seed: int, tasks: Iterable[str]) -> dict[str, DeltaSet]:
    paths = Paths(exp)
    excluded = {e.get("run_id") for e in read_exclusions(paths) if e.get("kind") == "run"}
    out = {}
    for t in tasks:
        d = paths.run_dir(base, t, seed)
        if d.name in excluded or not (d / "spectral" / "delta_factors.safetensors").exists():
            continue
        out[t] = DeltaSet.load(d / "spectral")
    return out


# ---------------------------------------------------------------------------
# Stage: evaluations
# ---------------------------------------------------------------------------


def _eval(exp: ExperimentConfig, base, task, eval_id: str, cache: dict, template: str | None = None, **kw) -> None:
    from .evaluate import run_evaluation

    paths = Paths(exp)
    if (paths.raw / f"{eval_id}.summary.json").exists():
        return
    s = run_evaluation(base, task, paths.raw, eval_id, split=exp.eval_split, template=template,
                       data_dir=exp.data_dir, max_examples=exp.eval_max_examples, device=exp.train.device,
                       dtype=exp.train.dtype, model_cache=cache, **kw)
    log(f"eval {eval_id}: {task.metric}={s['primary']:.4f} (n={s['n']})")


def _templates(exp: ExperimentConfig, task) -> list[str]:
    return [task.prompt_template_version, *exp.alt_templates]


def _eval_targets(exp: ExperimentConfig, task_id: str) -> list:
    """The task itself plus declared control tasks (for collateral damage)."""
    return [exp.task(task_id)] + [exp.task(c) for c in exp.control_tasks if c != task_id]


def eval_bases(exp: ExperimentConfig) -> None:
    for base in exp.bases:
        cache: dict = {}
        for task in exp.tasks:
            for tmpl in _templates(exp, task):
                _eval(exp, base, task, f"base__{base.name}__{task.task_id}__{tmpl}", cache, tmpl,
                      extra_meta={"role": "base", "task_id": task.task_id, "target_base": base.name,
                                  "eval_task": task.task_id})


def eval_direct(exp: ExperimentConfig) -> None:
    paths = Paths(exp)
    excluded = {e.get("run_id") for e in read_exclusions(paths) if e.get("kind") == "run"}
    for base in exp.bases:
        cache: dict = {}
        for task in exp.tasks:
            for seed in exp.seeds:
                run_dir = paths.run_dir(base.name, task.task_id, seed)
                if not (run_dir / "adapter").exists() or run_dir.name in excluded:
                    continue
                for ev_task in _eval_targets(exp, task.task_id):
                    for tmpl in _templates(exp, ev_task):
                        _eval(exp, base, ev_task,
                              f"direct__{base.name}__{task.task_id}__seed{seed}__on_{ev_task.task_id}__{tmpl}",
                              cache, tmpl, adapter_dir=run_dir / "adapter",
                              extra_meta={"role": "direct", "task_id": task.task_id, "eval_task": ev_task.task_id,
                                          "target_base": base.name, "seed": seed, "run_id": run_dir.name})


# ---------------------------------------------------------------------------
# Stage: predeclared gate + map fitting + prediction
# ---------------------------------------------------------------------------


def declare_gate(exp: ExperimentConfig) -> dict[str, Any]:
    """Freeze the gate before any held-out test runs; later edits are detected (spec §5.6)."""
    paths = Paths(exp)
    payload = {"gate": to_dict(exp.gate), "splits": [dataclasses.asdict(s) for s in exp.splits],
               "eval_split": exp.eval_split}
    digest = sha256_bytes(json.dumps(payload, sort_keys=True).encode())
    if paths.gate_declaration.exists():
        old = json.loads(paths.gate_declaration.read_text())
        if old["sha256"] != digest:
            raise RuntimeError(f"gate/splits changed after declaration ({paths.gate_declaration}); "
                               f"a changed gate needs a new experiment name")
        return old
    decl = {**payload, "sha256": digest, "declared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "code_commit": git_commit()}
    paths.gate_declaration.parent.mkdir(parents=True, exist_ok=True)
    paths.gate_declaration.write_text(json.dumps(decl, indent=1))
    return decl


def splits_of(exp: ExperimentConfig) -> list[TaskSplit]:
    ids = [t.task_id for t in exp.tasks]
    return [TaskSplit.from_holdout(s.split_id, ids, list(s.holdout)) for s in exp.splits]


def _base_topk(exp: ExperimentConfig, base_name: str) -> dict[str, dict[str, np.ndarray]]:
    from .alignment import discover_modules
    from .models import load_base_model

    paths = Paths(exp)
    f = paths.base_svd / f"{base_name}_k{exp.maps.k}.safetensors"
    if not f.exists():
        model, _, _ = load_base_model(exp.base(base_name), dtype=exp.train.dtype, device=exp.train.device)
        topk = base_weight_topk(model, discover_modules(model, exp.lora.target_modules), exp.maps.k)
        f.parent.mkdir(parents=True, exist_ok=True)
        save_file({f"{n}::{k}": np.ascontiguousarray(v, np.float32) for n, d in topk.items() for k, v in d.items()},
                  str(f))
        del model
    t = load_file(str(f))
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, v in t.items():
        n, k = key.rsplit("::", 1)
        out.setdefault(n, {})[k] = v.astype(np.float64)
    return out


def fit_maps(exp: ExperimentConfig, only_methods: list[str] | None = None) -> None:
    declare_gate(exp)
    paths = Paths(exp)
    task_ids = [t.task_id for t in exp.tasks]
    methods = _filter(exp.maps.methods, only_methods)
    for split in splits_of(exp):
        for src in exp.bases:
            for tgt in exp.bases:
                if src.name == tgt.name:
                    continue
                for seed in exp.seeds:
                    src_all = load_run_deltas(exp, src.name, seed, task_ids)
                    tgt_train = load_run_deltas(exp, tgt.name, seed, split.train)  # never loads held-out targets
                    src_train = {t: d for t, d in src_all.items() if t in split.train}
                    if not src_train or not tgt_train:
                        record_exclusion(paths, kind="map", reason="missing_training_runs", split=split.split_id,
                                         source=src.name, target=tgt.name, seed=seed)
                        continue
                    for method in methods:
                        _fit_and_predict(exp, paths, method, src, tgt, seed, split, src_all, src_train, tgt_train)


def _fit_and_predict(exp, paths, method, src, tgt, seed, split, src_all, src_train, tgt_train) -> None:
    map_dir = paths.maps / f"{src.name}__to__{tgt.name}__{split.split_id}__{method}__seed{seed}"
    if (map_dir / "map_meta.json").exists():
        pmap = PairMap.load(map_dir)
    else:
        base_svds = None
        if method == "cross_lora":
            any_task = next(iter(tgt_train))
            base_svds = cross_lora_params(_base_topk(exp, src.name), _base_topk(exp, tgt.name),
                                          src_train[any_task], tgt_train[any_task], exp.maps.layer_matching)
        pmap = fit_pair_map(method, src_train, tgt_train, split, exp.maps, src.name, tgt.name, base_svds, seed)
        pmap.save(map_dir)
    if not pmap.applicable:
        record_exclusion(paths, kind="map", reason="method_not_applicable", map_id=pmap.map_id, seed=seed,
                         note=next((m.note for m in pmap.modules.values() if m.note), ""))
        return
    for task_id in split.holdout:
        if task_id not in src_all:
            record_exclusion(paths, kind="prediction", reason="missing_source_run", map_id=pmap.map_id,
                             task_id=task_id, seed=seed)
            continue
        out = paths.preds / map_dir.name / task_id
        if (out / "delta_meta.json").exists():
            continue
        pred = pmap.predict(src_all[task_id])
        assert not pred.meta["in_sample"]
        pred.meta["family"] = tgt.family
        pred.save(out)
        # Diagnostic only (not used for fitting): compare with the real target update if it exists.
        true_dir = paths.run_dir(tgt.name, task_id, seed) / "spectral"
        if (true_dir / "delta_factors.safetensors").exists():
            rep = reconstruction_report(pred, DeltaSet.load(true_dir))
            rep.pop("modules", None)
            (out / "reconstruction.json").write_text(json.dumps(rep, indent=1))
    log(f"map {map_dir.name}: fitted on {len(src_train)} tasks, predicted {list(split.holdout)}")


def eval_transfer(exp: ExperimentConfig) -> None:
    paths = Paths(exp)
    if not paths.preds.exists():
        return
    by_target: dict[str, list[Path]] = {}
    for d in sorted(paths.preds.glob("*/*/delta_meta.json")):
        meta = json.loads(d.read_text())["meta"]
        by_target.setdefault(meta["target_base"], []).append(d.parent)
    for tgt_name, dirs in by_target.items():
        base = exp.base(tgt_name)
        cache: dict = {}
        for d in dirs:
            meta = json.loads((d / "delta_meta.json").read_text())["meta"]
            for ev_task in _eval_targets(exp, meta["task_id"]):
                for tmpl in _templates(exp, ev_task):
                    eval_id = (f"transfer__{meta['map_id']}__seed{meta['seed']}__{meta['task_id']}"
                               f"__on_{ev_task.task_id}__{tmpl}")
                    _eval(exp, base, ev_task, eval_id, cache, tmpl, delta_dir=d,
                          extra_meta={"role": "transfer", "task_id": meta["task_id"], "eval_task": ev_task.task_id,
                                      "method": meta["method"], "source_base": meta["source_base"],
                                      "target_base": meta["target_base"], "split_id": meta["split_id"],
                                      "seed": meta["seed"], "map_id": meta["map_id"],
                                      "train_task_ids": meta["train_task_ids"],
                                      "holdout_task_ids": meta["holdout_task_ids"]})


def run_all(exp: ExperimentConfig) -> None:
    from .analysis import analyze

    prepare_data(exp)
    train_grid(exp)
    eval_bases(exp)
    eval_direct(exp)
    if exp.splits:
        fit_maps(exp)
        eval_transfer(exp)
    analyze(exp)
