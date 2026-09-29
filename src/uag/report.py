"""Dry-run report: the four pre-rental checks, read from what the pipeline already recorded.

1. Did every run train without OOM / divergence, and how much GPU memory did it use?
2. Does the ΔW reconstruction check pass?
3. Does direct LoRA actually improve each task over the raw base?
4. How long do training and evaluation take (and a rough projection for Stage 0)?
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import yaml

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

    # 3: direct-LoRA lift
    summaries = [json.loads(p.read_text()) for p in sorted(paths.raw.glob("*.summary.json"))] if paths.raw.exists() else []
    canon = {t.task_id: t.prompt_template_version for t in exp.tasks}
    base = {(s["target_base"], s["task_id"]): s for s in summaries
            if s["role"] == "base" and s.get("template") == canon.get(s["task_id"])}
    direct = defaultdict(list)
    for s in summaries:
        if s["role"] == "direct" and s.get("eval_task") == s["task_id"] and s.get("template") == canon.get(s["task_id"]):
            direct[(s["target_base"], s["task_id"])].append(s)
    lines, n_ok, n_cells = [], 0, 0
    for b in exp.bases:
        for t in exp.tasks:
            sb, sd = base.get((b.name, t.task_id)), direct.get((b.name, t.task_id))
            if not sb or not sd:
                lines.append(f"       {b.name} / {t.task_id}: not evaluated yet")
                continue
            n_cells += 1
            d = float(np.mean([x["primary"] for x in sd]))
            lift = d - sb["primary"]
            n_ok += lift >= min_lift
            note = "" if lift >= min_lift else ("  <- no headroom (base near ceiling)" if sb["primary"] > 0.9
                                                  else "  <- adapter barely helps: more tokens / higher LR?")
            lines.append(f"       {b.name} / {t.task_id} ({t.metric}): base {sb['primary']:.3f} -> "
                         f"direct {d:.3f} (lift {lift:+.3f}){note}")
    ok_lift = n_cells > 0 and n_ok == n_cells
    out += ["", f"[{'PASS' if ok_lift else 'CHECK'}] 3. Direct LoRA improves the task (lift >= {min_lift}): "
                f"{n_ok}/{n_cells} cells", *lines]

    # 4: timing
    out += ["", "[INFO] 4. Timing on this machine"]
    by_base = defaultdict(list)
    for m, _ in runs:
        if m["tokens_seen"]:
            by_base[m["base_name"]].append((m["wall_seconds"] / m["tokens_seen"] * 1e6, m["total_params"]))
    for b, vals in by_base.items():
        sec_per_m = float(np.mean([v[0] for v in vals]))
        out.append(f"       training {b}: {sec_per_m / 60:.1f} min per 1M tokens "
                   f"({vals[0][1] / 1e9:.2f}B params incl. LoRA; includes validation passes)")
    ev = defaultdict(list)
    for s in summaries:
        if s.get("n"):
            ev[(s["target_base"], exp.task(s["eval_task"]).scoring)].append(s["eval_seconds"] / s["n"])
    for (b, scoring), v in sorted(ev.items()):
        out.append(f"       evaluation {b} [{scoring}]: {np.mean(v):.2f} s per example")

    if by_base and min(v[0][1] for v in by_base.values()) >= 3e8:
        # Very rough: time scales ~linearly with parameter count on the same GPU.
        rates = [(v[0][0] / v[0][1]) for v in by_base.values()]  # sec per 1M tokens per param
        per_param = float(np.mean(rates))
        stage0_params = [1.5e9, 3.1e9, 3.2e9, 2.6e9]
        train_h = sum(per_param * p * 2.0 * 10 * 3 for p in stage0_params) / 3600  # 2M tokens × 10 tasks × 3 seeds
        out += ["", f"       Rough Stage-0 training projection ON THIS GPU (linear in params, 120 runs × 2M tokens): "
                    f"~{train_h:.0f} GPU-hours.",
                "       An A40 is typically faster and needs no gradient checkpointing; treat this as an upper-end guide."]

    excl = read_exclusions(paths)
    if excl:
        out += ["", f"[CHECK] Exclusion log has {len(excl)} entries: {paths.exclusions}"]
    verdict = ok_train and ok_delta and ok_lift
    out += ["", "Overall: " + ("ready to rent — go to the pilot." if verdict else
                               "fix the CHECK items above first (paste this report to Claude if unsure).")]
    return "\n".join(out)
