"""Dry-run report: the four pre-rental checks, read from what the pipeline already recorded.

1. Did every run train without OOM / divergence, and how much GPU memory did it use?
2. Does the ΔW reconstruction check pass?
3. Does direct LoRA improve each task over the *few-shot* base, with headroom left (no base at
   ceiling) and more than just the output format learned?
4. How long do training and evaluation take on this machine? (Rental cost comes from
   ``scripts/benchmark_gpu.sh`` on the rented card, not from extrapolating this machine.)
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import yaml

from .analysis import CEILING, FORMAT_SHARE, headroom_rows, load_summaries
from .config import ExperimentConfig
from .pipeline import Paths, read_exclusions


def _gb(x) -> str:
    return f"{x / 1e9:.1f} GB" if x else "n/a (CPU)"


def dry_run_report(exp: ExperimentConfig, min_lift: float = 0.05) -> str:
    paths = Paths(exp)
    out: list[str] = [f"Dry-run report for experiment '{exp.name}'", ""]
    expected = [(b.name, t.task_id, s) for b in exp.bases for t in exp.tasks for s in exp.seeds]

    # 1 + 2: runs, memory, divergence, ΔW verification
    runs, missing = [], []
    for b, t, s in expected:
        d = paths.run_dir(b, t, s)
        if not (d / "manifest.yaml").exists():
            missing.append(d.name)
            continue
        m = yaml.safe_load((d / "manifest.yaml").read_text())
        v = json.loads((d / "delta_verification.json").read_text()) if (d / "delta_verification.json").exists() else None
        runs.append((m, v))
    ok_train = not missing and all(not m.get("diverged") for m, _ in runs)
    out.append(f"[{'PASS' if ok_train else 'CHECK'}] 1. Training finished: {len(runs)}/{len(expected)} runs")
    for name in missing:
        out.append(f"       missing (crashed / OOM / not run yet): {name}")
    for m, _ in runs:
        flag = "  DIVERGED" if m.get("diverged") else ""
        out.append(f"       {m['run_id']}: peak VRAM {_gb(m.get('peak_vram_bytes'))}, "
                   f"{m['wall_seconds'] / 60:.1f} min, {m['tokens_seen']:,} tokens, "
                   f"best {m['selection']['metric']}={m['selection']['value']:.4f} "
                   f"(start {m['selection']['initial'].get('valid_loss', float('nan')):.4f}){flag}")

    verified = [(m, v) for m, v in runs if v is not None]
    ok_delta = bool(verified) and all(v["passed"] for _, v in verified) and len(verified) == len(runs)
    out += ["", f"[{'PASS' if ok_delta else 'CHECK'}] 2. ΔW reconstruction check: "
                f"{sum(v['passed'] for _, v in verified)}/{len(runs)} runs pass"]
    for m, v in verified:
        out.append(f"       {m['run_id']}: max logit diff {v['max_abs_logit_diff']:.2e} "
                   f"(tolerance {v.get('tolerance', v.get('atol', float('nan'))):.2e}), "
                   f"adapter effect {v['adapter_effect_max_abs']:.2e}")

    # 3: direct-LoRA lift over the declared baseline, with ceiling / format flags
    summaries = load_summaries(paths)
    lines, n_ok, n_cells = [], 0, 0
    ref = exp.gate.baseline
    for r in headroom_rows(exp, summaries):
        name = f"{r['base']} / {r['task']} ({r['metric']})"
        if r["s_base"] is None or r["n_seeds"] == 0:
            lines.append(f"       {name}: not evaluated yet")
            continue
        n_cells += 1
        problems = []
        if r["ceiling"]:
            problems.append(f"CEILING: a base already scores >= {CEILING}; no room to measure transfer")
        if r["format_dominated"]:
            problems.append(f"FORMAT: {r['format_share']:.0%} of the lift comes from showing examples, "
                            f"not from the adapter")
        if r["direct_lift"] < min_lift:
            problems.append(f"lift vs {ref} base below {min_lift}")
        n_ok += not problems
        zs = "–" if r["s_base_zeroshot"] is None else f"{r['s_base_zeroshot']:.3f}"
        fs = "–" if r["s_base_fewshot"] is None else f"{r['s_base_fewshot']:.3f}"
        lines.append(f"       {name}: zero-shot {zs}, {exp.fewshot.k}-shot {fs} -> direct {r['s_direct_mean']:.3f} "
                     f"(lift vs {ref} {r['direct_lift']:+.3f})")
        lines += [f"         <- {p}" for p in problems]
    ok_lift = n_cells > 0 and n_ok == n_cells
    out += ["", f"[{'PASS' if ok_lift else 'CHECK'}] 3. Direct LoRA beats the {ref} base by >= {min_lift}, "
                f"no base >= {CEILING}, lift not format-dominated (few-shot share < {FORMAT_SHARE:.0%}): "
                f"{n_ok}/{n_cells} cells", *lines]

    # Geometry sanity: does ΔW reflect the task or the random init?
    from .diagnostics import render_geometry_checks

    out += ["", "[INFO] Geometry sanity (is ΔW task signal or random-init noise?)",
            *["       " + ln for ln in render_geometry_checks(exp)]]

    # 4: timing
    out += ["", "[INFO] 4. Timing on this machine (not a rental projection: run scripts/benchmark_gpu.sh "
                "on the rented GPU for that)"]
    for m, _ in runs:
        if m["steps"]:
            tps = m["tokens_seen"] / m["steps"]
            out.append(f"       {m['run_id']}: {m['wall_seconds'] / m['steps']:.2f} s/step, "
                       f"{tps:,.0f} tokens/step, {m['tokens_seen'] / m['wall_seconds']:,.0f} tokens/s "
                       f"(includes validation passes)")
    ev = defaultdict(list)
    for s in summaries:
        if s.get("n"):
            ev[(s["target_base"], exp.task(s["eval_task"]).scoring, s.get("shots", 0))].append(
                s["eval_seconds"] / s["n"])
    for (b, scoring, shots), v in sorted(ev.items()):
        out.append(f"       evaluation {b} [{scoring}, {shots}-shot]: {np.mean(v):.2f} s per example")

    excl = read_exclusions(paths)
    if excl:
        out += ["", f"[CHECK] Exclusion log has {len(excl)} entries: {paths.exclusions}"]
    verdict = ok_train and ok_delta and ok_lift
    out += ["", "Overall: " + ("ready to rent — go to the pilot." if verdict else
                               "fix the CHECK items above first (paste this report to Claude if unsure).")]
    return "\n".join(out)
