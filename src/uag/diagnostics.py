"""Is the measured geometry task signal or random-init noise? (pre-Stage-0 sanity checks)

LoRA starts with a random A (drawn from the run's seed) and B = 0. If a task is learned in a
few dozen steps, A may barely move, and ΔW = scale·BA then acts on inputs through directions
that were fixed by the random init. Two checks answer this from the trained runs:

1. **Factor movement** (recorded per run by training): ||A - A0|| / ||A0||, ||B||, and how much
   A's row space still overlaps A0's.
2. **Seed comparison** on one base, using :func:`uag.geometry.seed_invariance`-style pairwise
   similarity in three groups:
   - same task, different seeds: task signal, if the geometry is not random;
   - different tasks, same seed: shares the init A0 (under ``init_seed_scope: seed``);
   - different tasks, different seeds: the reference level.
   Same-task agreement near the reference means ΔW is mostly seed noise. Same-seed agreement
   well above the reference means the shared init, not the task, sets part of the geometry.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import numpy as np
import yaml

from .config import ExperimentConfig
from .extract_delta import DeltaSet
from .geometry import deltaset_similarity, seed_invariance
from .pipeline import Paths, _filter, load_run_deltas

METRICS = ("global_cosine", "right_overlap", "left_overlap")


def chance_overlap(ds: DeltaSet) -> dict[str, float]:
    """Expected subspace overlap of two unrelated rank-r subspaces: ~ r / d per side."""
    left = [lr.rank / lr.shape[0] for lr in ds.modules.values() if lr.rank]
    right = [lr.rank / lr.shape[1] for lr in ds.modules.values() if lr.rank]
    return {"left_overlap": float(np.mean(left)) if left else float("nan"),
            "right_overlap": float(np.mean(right)) if right else float("nan"), "global_cosine": 0.0}


def _mean_sims(pairs: list[tuple[DeltaSet, DeltaSet]]) -> dict[str, Any]:
    sims = [deltaset_similarity(a, b) for a, b in pairs]
    if not sims:
        return {"n_pairs": 0}
    return {"n_pairs": len(sims), **{m: float(np.mean([s[m] for s in sims])) for m in METRICS}}


def seed_comparison(exp: ExperimentConfig, only_bases: list[str] | None = None,
                    only_tasks: list[str] | None = None) -> dict[str, Any]:
    """Pairwise ΔW similarity per base in the three groups above (plus per-task seed invariance)."""
    out: dict[str, Any] = {"bases": {}}
    tasks = [t.task_id for t in _filter(exp.tasks, only_tasks, lambda t: t.task_id)]
    for base in _filter(exp.bases, only_bases, lambda b: b.name):
        runs: dict[tuple[str, int], DeltaSet] = {}
        for seed in exp.seeds:
            for t, ds in load_run_deltas(exp, base.name, seed, tasks).items():
                runs[(t, seed)] = ds
        if len(runs) < 2:
            continue
        same_task, same_seed, neither = [], [], []
        for (ka, a), (kb, b) in itertools.combinations(sorted(runs.items()), 2):
            if ka[0] == kb[0]:
                same_task.append((a, b))
            elif ka[1] == kb[1]:
                same_seed.append((a, b))
            else:
                neither.append((a, b))
        per_task = {}
        for t in tasks:
            seeds = [runs[(t, s)] for s in exp.seeds if (t, s) in runs]
            if len(seeds) >= 2:
                inv = seed_invariance(seeds)
                per_task[t] = {"n_pairs": inv["n_pairs"], **{m: inv[m]["mean"] for m in METRICS}}
        out["bases"][base.name] = {
            "runs": [f"{t}/seed{s}" for t, s in sorted(runs)],
            "chance": chance_overlap(next(iter(runs.values()))),
            "same_task_diff_seed": _mean_sims(same_task),
            "diff_task_same_seed": _mean_sims(same_seed),
            "diff_task_diff_seed": _mean_sims(neither),
            "per_task_seed_invariance": per_task,
            "init_seed_scope": exp.lora.init_seed_scope,
        }
    return out


def factor_movement_rows(exp: ExperimentConfig, only_bases: list[str] | None = None,
                         only_tasks: list[str] | None = None) -> list[dict[str, Any]]:
    paths = Paths(exp)
    rows = []
    for b in _filter(exp.bases, only_bases, lambda x: x.name):
        for t in _filter(exp.tasks, only_tasks, lambda x: x.task_id):
            for s in exp.seeds:
                f = paths.run_dir(b.name, t.task_id, s) / "manifest.yaml"
                if not f.exists():
                    continue
                m = yaml.safe_load(f.read_text())
                fm = (m.get("factor_movement") or {}).get("overall")
                if fm:
                    rows.append({"run_id": m["run_id"], **fm})
    return rows


def _f(x: float | None, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def render_geometry_checks(exp: ExperimentConfig, only_bases: list[str] | None = None,
                           only_tasks: list[str] | None = None) -> list[str]:
    """Plain-text section shared by ``uag seed-compare`` and the dry-run report."""
    L: list[str] = []
    rows = factor_movement_rows(exp, only_bases, only_tasks)
    L.append("Factor movement (selected checkpoint vs initialisation):")
    if not rows:
        L.append("  no runs with recorded factor movement (runs trained before this was added?)")
    for r in rows:
        L.append(f"  {r['run_id']}: ||A-A0||/||A0|| = {_f(r['a_rel_move_mean'])} "
                 f"(min {_f(r['a_rel_move_min'])}, max {_f(r['a_rel_move_max'])}), "
                 f"||B||/||A0|| = {_f(r['b_rel_to_a0_mean'])}, "
                 f"row-space overlap with A0 = {_f(r['a_rowspace_overlap_mean'])} "
                 f"(chance {_f(r['chance_overlap_mean'], 4)})")
    if rows:
        worst = max(r["a_rowspace_overlap_mean"] for r in rows)
        if worst > 0.9:
            L.append("  -> A's input directions are still >90% those of its random init in at least one run: "
                     "ΔW's input side is mostly set by the seed, not the task.")

    res = seed_comparison(exp, only_bases, only_tasks)
    L += ["", "Seed comparison (pairwise ΔW similarity; module-averaged subspace overlaps, global cosine):"]
    if not res["bases"]:
        L.append("  needs >= 2 trained runs on one base")
    for b, r in res["bases"].items():
        ch = r["chance"]
        L.append(f"  {b} (init_seed_scope: {r['init_seed_scope']}; runs {', '.join(r['runs'])})")
        L.append(f"    {'group':<28}{'pairs':>6}{'cosine':>9}{'in-side':>9}{'out-side':>9}")
        for label, key in (("same task, diff seed", "same_task_diff_seed"),
                           ("diff task, same seed", "diff_task_same_seed"),
                           ("diff task, diff seed", "diff_task_diff_seed")):
            g = r[key]
            if g["n_pairs"]:
                L.append(f"    {label:<28}{g['n_pairs']:>6}{g['global_cosine']:>9.3f}"
                         f"{g['right_overlap']:>9.3f}{g['left_overlap']:>9.3f}")
        L.append(f"    {'chance (unrelated)':<28}{'':>6}{0:>9.3f}{ch['right_overlap']:>9.3f}{ch['left_overlap']:>9.3f}")
        for t, v in r["per_task_seed_invariance"].items():
            L.append(f"    seed invariance {t}: cosine {v['global_cosine']:.3f}, in-side {v['right_overlap']:.3f}, "
                     f"out-side {v['left_overlap']:.3f} ({v['n_pairs']} pairs)")
        st, ss, nn = r["same_task_diff_seed"], r["diff_task_same_seed"], r["diff_task_diff_seed"]
        ref = nn if nn["n_pairs"] else None
        if st["n_pairs"] and ref:
            if st["global_cosine"] < 2 * max(ref["global_cosine"], 0.02):
                L.append("    -> same-task runs agree barely more than unrelated runs: the geometry looks "
                         "seed-dominated at this budget.")
            else:
                L.append("    -> same-task runs agree clearly more than unrelated runs: there is task signal.")
        if ss["n_pairs"] and ref and ss["right_overlap"] > 2 * ref["right_overlap"]:
            L.append("    -> different tasks that share a seed share input directions (shared random A0). "
                     "Use lora.init_seed_scope: task so held-out tasks don't inherit geometry from the init.")
        elif ss["n_pairs"] and not ref:
            L.append("    (train a second seed to separate shared-init effects from task effects)")
    return L


def save_seed_comparison(exp: ExperimentConfig, only_bases=None, only_tasks=None) -> str:
    paths = Paths(exp)
    res = seed_comparison(exp, only_bases, only_tasks)
    res["factor_movement"] = factor_movement_rows(exp, only_bases, only_tasks)
    out = paths.exp_results / "seed_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return str(out)
