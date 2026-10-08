"""Learning curves → token budgets (spec §5.3: identical budgets across bases).

How the budget is set:

1. **Pilot** each task with early stopping and a generous cap (``max_tokens_seen``), on every
   base (or the dev bases first).
2. For every run, read the validation curve from ``train_log.jsonl`` and find ``t95``: the
   tokens seen when the selection metric first reached 95% of its total improvement.
3. The Stage-0 **cap per task** is ``CAP_FACTOR`` × the largest ``t95`` over all bases and
   seeds of that task, rounded up. The same cap and the same early-stopping rule then apply to
   every base, so budgets stay identical across bases; the run manifest records where each run
   actually stopped and which checkpoint was selected.

A run whose best checkpoint is at the very end of its budget (``cap_binding``) was still
improving: raise that task's cap before trusting its t95.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import yaml

from .config import ExperimentConfig
from .pipeline import Paths, _filter

CAP_FACTOR = 2.0
FRACTIONS = (0.5, 0.9, 0.95)


def _nice_ceiling(x: float) -> int:
    """Round up to 1, 2 or 5 × a power of ten (e.g. 137k -> 200k)."""
    if x <= 0:
        return 0
    p = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if x <= m * p:
            return int(m * p)
    return int(10 * p)


def curve(run_dir, metric: str) -> list[tuple[int, int, float]]:
    """(step, tokens_seen, value) for every validation evaluation, step 0 included."""
    f = run_dir / "train_log.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return [(r["step"], r["tokens_seen"], r[metric]) for r in rows if metric in r]


def curve_summary(points: list[tuple[int, int, float]], lower_is_better: bool) -> dict[str, Any]:
    if len(points) < 2:
        return {}
    sign = -1.0 if lower_is_better else 1.0
    v0 = sign * points[0][2]
    best_i = max(range(len(points)), key=lambda i: (sign * points[i][2], -i))
    total = sign * points[best_i][2] - v0
    out: dict[str, Any] = {"initial": points[0][2], "best": points[best_i][2], "best_step": points[best_i][0],
                           "best_tokens": points[best_i][1], "last_tokens": points[-1][1],
                           "cap_binding": best_i == len(points) - 1 and len(points) > 2}
    for frac in FRACTIONS:
        key = f"t{int(frac * 100)}"
        out[key] = None
        if total <= 0:
            continue
        for _, tokens, v in points:
            if sign * v - v0 >= frac * total:
                out[key] = tokens
                break
    return out


def learning_curves(exp: ExperimentConfig, only_bases: list[str] | None = None,
                    only_tasks: list[str] | None = None) -> dict[str, Any]:
    paths = Paths(exp)
    metric = exp.train.selection_metric
    rows = []
    for b in _filter(exp.bases, only_bases, lambda x: x.name):
        for t in _filter(exp.tasks, only_tasks, lambda x: x.task_id):
            lower = metric == "valid_loss" or t.metric_direction == "lower"
            for s in exp.seeds:
                d = paths.run_dir(b.name, t.task_id, s)
                if not (d / "manifest.yaml").exists():
                    continue
                m = yaml.safe_load((d / "manifest.yaml").read_text())
                summ = curve_summary(curve(d, metric), lower)
                rows.append({"run_id": m["run_id"], "base": b.name, "task": t.task_id, "seed": s, "metric": metric,
                             "stop_reason": (m.get("stopping") or {}).get("reason", "token_cap (legacy run)"),
                             "stop_tokens": m["tokens_seen"], "steps": m["steps"],
                             "tokens_per_step": m["tokens_seen"] / max(m["steps"], 1),
                             "wall_seconds": m["wall_seconds"], **summ})
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    proposals = {}
    for task, rs in by_task.items():
        t95 = [r["t95"] for r in rs if r.get("t95")]
        binding = [r["run_id"] for r in rs if r.get("cap_binding")]
        proposals[task] = {"max_t95_tokens": max(t95) if t95 else None,
                           "proposed_cap_tokens": _nice_ceiling(CAP_FACTOR * max(t95)) if t95 else None,
                           "n_runs": len(rs), "cap_binding_runs": binding,
                           "no_improvement_runs": [r["run_id"] for r in rs if not r.get("t95")]}
    return {"metric": metric, "cap_factor": CAP_FACTOR, "runs": rows, "proposals": proposals}


def render_learning_curves(res: dict[str, Any]) -> list[str]:
    def tok(x):
        return "–" if x is None else f"{x / 1000:,.0f}k"

    L = [f"Learning curves on {res['metric']} (tokens until the run first reached 50/90/95% of its total "
         f"improvement)", f"  {'run':<52}{'start':>8}{'best':>8}{'t50':>8}{'t90':>8}{'t95':>8}{'stopped':>10}  reason"]
    for r in res["runs"]:
        if "initial" not in r:
            L.append(f"  {r['run_id']:<52} (no validation curve)")
            continue
        flag = "  <- still improving at the end: raise the cap" if r["cap_binding"] else ""
        L.append(f"  {r['run_id']:<52}{r['initial']:>8.3f}{r['best']:>8.3f}{tok(r['t50']):>8}{tok(r['t90']):>8}"
                 f"{tok(r['t95']):>8}{tok(r['stop_tokens']):>10}  {r['stop_reason']}{flag}")
    L += ["", f"Proposed per-task token cap = {res['cap_factor']:g} x the largest t95 over bases and seeds "
              f"(rounded up); the same cap applies to every base:"]
    for task, p in sorted(res["proposals"].items()):
        extra = []
        if p["cap_binding_runs"]:
            extra.append(f"{len(p['cap_binding_runs'])} run(s) were still improving at their cap, so this is a lower bound")
        if p["no_improvement_runs"]:
            extra.append(f"{len(p['no_improvement_runs'])} run(s) never improved")
        L.append(f"  {task:<20} max t95 {tok(p['max_t95_tokens']):>8} -> cap {tok(p['proposed_cap_tokens']):>8}"
                 + (f"   ({'; '.join(extra)})" if extra else ""))
    return L
