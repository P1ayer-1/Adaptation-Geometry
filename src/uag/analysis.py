"""Frozen Stage-0 analysis: regenerates every table and the go/no-go memo from raw results.

Inputs are only the ``*.summary.json`` / ``*.jsonl`` files under ``results/raw/<experiment>``
plus the predeclared gate (``results/<experiment>/gate_declaration.json``). Outputs:

- ``results/tables/<experiment>_cells.csv``     every target×task×seed×method transfer cell
- ``results/tables/<experiment>_direct.csv``    raw-base and direct-LoRA headroom table
- ``results/<experiment>_decision.md``          the go/no-go memo (spec §5.6, §20.10)
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .metrics import paired_bootstrap_diff, recovered_lift, seed_summary
from .pipeline import Paths, declare_gate, read_exclusions
from .provenance import git_commit


def load_summaries(paths: Paths) -> list[dict[str, Any]]:
    if not paths.raw.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(paths.raw.glob("*.summary.json"))]


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


# Headroom flags (reported in the memo and enforced by the dry-run report).
CEILING = 0.9        # a base (zero- or few-shot) at or above this leaves no room to measure transfer
FORMAT_SHARE = 0.5   # few-shot gain >= this share of the zero-shot-referenced direct lift: format-dominated


def base_scores(exp: ExperimentConfig, summaries: list[dict[str, Any]], template: str | None = None):
    """Zero-shot and few-shot base scores keyed by (target, task), for one template."""
    zs, fs = {}, {}
    for s in summaries:
        tmpl = template or exp.task(s["task_id"]).prompt_template_version
        if s.get("template") != tmpl or s.get("eval_task", s["task_id"]) != s["task_id"]:
            continue
        if s["role"] == "base":
            zs[(s["target_base"], s["task_id"])] = s["primary"]
        elif s["role"] == "base_fewshot" and s.get("shots") == exp.fewshot.k \
                and s.get("fewshot_seed") == exp.fewshot.seed:
            fs[(s["target_base"], s["task_id"])] = s["primary"]
    return zs, fs


def reference_scores(exp: ExperimentConfig, zs: dict, fs: dict) -> dict:
    """S_base used for lift / eligibility / RecoveredLift, per the declared ``gate.baseline``."""
    return fs if exp.gate.baseline == "fewshot" else zs


def build_tables(exp: ExperimentConfig, summaries: list[dict[str, Any]], template: str | None = None):
    """Index scores: base[(target, task)] (the reference baseline), direct[(target, task, seed)],
    transfer cells. Cells whose reference baseline was not evaluated are skipped."""
    higher = {t.task_id: t.metric_direction == "higher" for t in exp.tasks}
    zs, fs = base_scores(exp, summaries, template)
    base = reference_scores(exp, zs, fs)
    direct, transfer = {}, []
    for s in summaries:
        tmpl = template or exp.task(s["task_id"]).prompt_template_version
        if s.get("template") != tmpl or s.get("eval_task", s["task_id"]) != s["task_id"]:
            continue  # on-task, canonical-template scores only (controls handled separately)
        if s["role"] == "direct":
            direct[(s["target_base"], s["task_id"], s["seed"])] = s["primary"]
        elif s["role"] == "transfer":
            transfer.append(s)
    cells = []
    for s in transfer:
        key_b = (s["target_base"], s["task_id"])
        key_d = (s["target_base"], s["task_id"], s["seed"])
        if key_b not in base or key_d not in direct:
            continue
        direct_seeds = [v for (b, t, _), v in direct.items() if (b, t) == key_b]
        mean_direct = float(np.mean(direct_seeds))
        eligible = recovered_lift(mean_direct, base[key_b], mean_direct, exp.gate.min_direct_lift,
                                  higher_is_better=higher[s["task_id"]]).eligible
        lift = recovered_lift(s["primary"], base[key_b], direct[key_d], exp.gate.min_direct_lift,
                              higher_is_better=higher[s["task_id"]])
        src_fam = exp.base(s["source_base"]).family
        tgt_fam = exp.base(s["target_base"]).family
        cells.append({
            "source": s["source_base"], "target": s["target_base"], "heterogeneous": src_fam != tgt_fam,
            "task": s["task_id"], "split": s["split_id"], "seed": s["seed"], "method": s["method"],
            "s_base": base[key_b], "s_base_zeroshot": zs.get(key_b), "s_base_fewshot": fs.get(key_b),
            "s_direct": direct[key_d], "s_transfer": s["primary"],
            "transfer_lift": lift.transfer_lift, "direct_lift": lift.direct_lift,
            "recovered_lift": lift.recovered_lift if eligible else None, "capped": lift.capped,
            "eligible": eligible, "eval_id": s["eval_id"],
        })
    return base, direct, cells


def headroom_rows(exp: ExperimentConfig, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per base × task: zero-shot, few-shot and direct scores, lift vs the declared
    baseline, and the ceiling / format-dominated flags."""
    zs, fs = base_scores(exp, summaries)
    ref = reference_scores(exp, zs, fs)
    _, direct, _ = build_tables(exp, summaries)
    rows = []
    for b in exp.bases:
        for t in exp.tasks:
            key = (b.name, t.task_id)
            sign = 1 if t.metric_direction == "higher" else -1
            ds = [v for (bb, tt, _), v in direct.items() if (bb, tt) == key]
            ss = seed_summary(ds)
            s_zs, s_fs, s_ref = zs.get(key), fs.get(key), ref.get(key)
            lift = sign * (ss["mean"] - s_ref) if ds and s_ref is not None else None
            lift_zs = sign * (ss["mean"] - s_zs) if ds and s_zs is not None else None
            fs_gain = sign * (s_fs - s_zs) if s_fs is not None and s_zs is not None else None
            format_share = (fs_gain / lift_zs if fs_gain is not None and lift_zs is not None
                            and lift_zs >= exp.gate.min_direct_lift else None)
            known = [x for x in (s_zs, s_fs) if x is not None]
            ceiling = sign > 0 and bool(known) and max(known) >= CEILING
            rows.append({"base": b.name, "family": b.family, "task": t.task_id, "metric": t.metric,
                         "s_base_zeroshot": s_zs, "s_base_fewshot": s_fs, "baseline": exp.gate.baseline,
                         "s_base": s_ref, "s_direct_mean": ss["mean"], "s_direct_std": ss["std"],
                         "n_seeds": ss["n"], "direct_lift": lift, "direct_lift_vs_zeroshot": lift_zs,
                         "format_share": format_share, "ceiling": ceiling,
                         "format_dominated": format_share is not None and format_share >= FORMAT_SHARE,
                         "eligible": lift is not None and lift >= exp.gate.min_direct_lift})
    return rows


def per_example_scores(paths: Paths, eval_id: str) -> list[float]:
    f = paths.raw / f"{eval_id}.jsonl"
    return [json.loads(l)["score"] for l in f.read_text().splitlines()] if f.exists() else []


def graded_criterion(exp: ExperimentConfig, graded_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Primary method vs the strongest non-learned baseline on RecoveredNLL, paired over identical
    source×target×task cells (seeds and splits averaged)."""
    g = exp.gate
    if not graded_rows:
        return {"value": None, "passed": False, "note": "graded transfer not computed (run `uag graded-transfer`)"}

    def cell_means(m: str) -> dict[tuple, float]:
        acc = defaultdict(list)
        for r in graded_rows:
            if r["method"] == m and r.get("recovered_nll") is not None:
                acc[(r["source"], r["target"], r["task"])].append(r["recovered_nll"])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    learned = cell_means(g.primary_method)
    base = {b: cell_means(b) for b in g.non_learned_baselines}
    base = {b: v for b, v in base.items() if set(v) & set(learned)}
    if not learned or not base:
        return {"value": None, "passed": False, "note": "no comparable graded cells"}
    means = {b: float(np.mean([v[k] for k in set(v) & set(learned)])) for b, v in base.items()}
    strongest = max(means, key=means.get)
    common = sorted(set(base[strongest]) & set(learned))
    diff = paired_bootstrap_diff([learned[k] for k in common], [base[strongest][k] for k in common],
                                 n_boot=g.bootstrap_samples, level=g.ci_level, seed=g.seed)
    return {"value": diff, "passed": diff["low"] > 0, "strongest_baseline": strongest,
            "mean_recovered_nll": {g.primary_method: float(np.mean([learned[k] for k in common])), **means}}


def evaluate_gate(exp: ExperimentConfig, cells: list[dict[str, Any]],
                  graded_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    g = exp.gate
    method = g.primary_method
    elig = [c for c in cells if c["eligible"]]
    out: dict[str, Any] = {"primary_method": method, "criteria": {}}

    # (1) heterogeneous ordered pairs with positive, seed-repeatable held-out transfer on a
    #     majority of eligible tasks.
    pair_tasks: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for c in elig:
        if c["method"] == method and c["heterogeneous"]:
            pair_tasks[(c["source"], c["target"])][c["task"]].append(c["transfer_lift"])
    pair_rows = []
    for (s, t), tasks in sorted(pair_tasks.items()):
        pos = {task: (np.mean(v) > 0 and np.mean(np.array(v) > 0) > 0.5) for task, v in tasks.items()}
        n_pos = sum(pos.values())
        pair_rows.append({"source": s, "target": t, "n_eligible_tasks": len(tasks), "n_positive": n_pos,
                          "passes": n_pos > len(tasks) / 2})
    n_pass = sum(r["passes"] for r in pair_rows)
    out["pairs"] = pair_rows
    out["criteria"]["heterogeneous_pairs"] = {"value": n_pass, "threshold": g.min_heterogeneous_pairs,
                                              "passed": n_pass >= g.min_heterogeneous_pairs}

    # (2) median RecoveredLift over eligible held-out cells (averaged over seeds per cell).
    rl = defaultdict(list)
    for c in elig:
        if c["method"] == method and c["recovered_lift"] is not None:
            rl[(c["source"], c["target"], c["task"], c["split"])].append(c["recovered_lift"])
    med = float(np.median([np.mean(v) for v in rl.values()])) if rl else float("nan")
    out["criteria"]["median_recovered_lift"] = {"value": med, "threshold": g.min_median_recovered_lift,
                                                "passed": bool(rl) and med >= g.min_median_recovered_lift}

    # (3) learned map beats the strongest non-learned baseline: paired bootstrap over identical
    #     source×target×task cells (seed/split averaged), 95% CI excluding zero (and positive).
    def cell_means(m: str) -> dict[tuple, float]:
        acc = defaultdict(list)
        for c in elig:
            if c["method"] == m:
                acc[(c["source"], c["target"], c["task"])].append(c["s_transfer"])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    learned = cell_means(method)
    comps = []
    for b in g.non_learned_baselines:
        bm = cell_means(b)
        common = sorted(set(learned) & set(bm))
        if not common:
            comps.append({"baseline": b, "n": 0, "mean_score": None})
            continue
        comps.append({"baseline": b, "n": len(common), "mean_score": float(np.mean([bm[k] for k in common])),
                      "common": common})
    comps_valid = [c for c in comps if c["n"] > 0]
    out["baselines"] = [{k: v for k, v in c.items() if k != "common"} for c in comps]
    if comps_valid and learned:
        strongest = max(comps_valid, key=lambda c: c["mean_score"])
        common = strongest["common"]
        bm = cell_means(strongest["baseline"])
        diff = paired_bootstrap_diff([learned[k] for k in common], [bm[k] for k in common],
                                     n_boot=g.bootstrap_samples, level=g.ci_level, seed=g.seed)
        out["strongest_baseline"] = strongest["baseline"]
        out["criteria"]["beats_strongest_baseline"] = {"value": diff, "passed": diff["low"] > 0}
    else:
        out["strongest_baseline"] = None
        out["criteria"]["beats_strongest_baseline"] = {"value": None, "passed": False}
    if g.graded_beats_baselines:
        out["criteria"]["graded_beats_baselines"] = graded_criterion(exp, graded_rows)
    out["passed"] = all(c["passed"] for c in out["criteria"].values())
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _reconstruction_rows(paths: Paths) -> dict[tuple, dict[str, float]]:
    out = {}
    for f in paths.preds.glob("*/*/reconstruction.json") if paths.preds.exists() else []:
        meta = json.loads((f.parent / "delta_meta.json").read_text())["meta"]
        out[(meta["map_id"], meta["seed"], meta["task_id"])] = json.loads(f.read_text())
    return out


def analyze(exp: ExperimentConfig) -> dict[str, Any]:
    paths = Paths(exp)
    decl = declare_gate(exp) if exp.splits else None  # raises if the gate was edited after declaration
    summaries = load_summaries(paths)
    _, _, cells = build_tables(exp, summaries)
    direct_rows = headroom_rows(exp, summaries)
    _write_csv(paths.tables / f"{exp.name}_direct.csv", direct_rows)
    _write_csv(paths.tables / f"{exp.name}_cells.csv", [{k: v for k, v in c.items()} for c in cells])

    graded_file = paths.exp_results / "graded_transfer.json"
    graded_rows = json.loads(graded_file.read_text())["rows"] if graded_file.exists() else None
    gate = evaluate_gate(exp, cells, graded_rows) if cells else None
    recon = _reconstruction_rows(paths)
    md = render_memo(exp, paths, decl, direct_rows, cells, gate, recon, summaries)
    paths.decision.parent.mkdir(parents=True, exist_ok=True)
    paths.decision.write_text(md)
    result = {"decision_path": str(paths.decision), "n_cells": len(cells), "gate": gate}
    (paths.exp_results / "analysis.json").parent.mkdir(parents=True, exist_ok=True)
    (paths.exp_results / "analysis.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def render_memo(exp, paths, decl, direct_rows, cells, gate, recon, summaries) -> str:
    L: list[str] = []
    title = "Stage-0 decision memo" if exp.name == "stage0" else f"Decision memo — experiment `{exp.name}`"
    L += [f"# {title}", "",
          f"_Generated automatically by `uag analyze` (src/uag/analysis.py) on "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')} at commit `{git_commit()['commit'][:12]}`. "
          f"Do not edit by hand; rerun the analysis instead._", ""]
    if any(b.source == "tiny" for b in exp.bases):
        L += ["> **Pipeline validation only.** This experiment uses randomly initialised tiny bases; "
              "its numbers carry no scientific weight.", ""]
    L += ["## Setup", "",
          f"- Bases: " + ", ".join(f"`{b.name}` ({b.family}, rev `{(b.revision or 'local')[:12]}`)" for b in exp.bases),
          f"- Tasks: " + ", ".join(f"`{t.task_id}`" for t in exp.tasks),
          f"- Seeds: {exp.seeds}; LoRA r={exp.lora.rank}, alpha={exp.lora.alpha}, modules={exp.lora.target_modules}",
          f"- Evaluation split: `{exp.eval_split}`" + (f" (first {exp.eval_max_examples} examples)" if exp.eval_max_examples else ""),
          f"- Held-out splits: " + "; ".join(f"`{s.split_id}` → {s.holdout}" for s in exp.splits), ""]
    if decl:
        g = decl["gate"]
        L += ["## Predeclared gate", "",
              f"Declared {decl['declared_at']} (sha256 `{decl['sha256'][:12]}`), before held-out evaluation.", "",
              f"1. At least **{g['min_heterogeneous_pairs']}** heterogeneous ordered base pairs show positive, "
              f"seed-repeatable held-out transfer on a majority of eligible tasks;",
              f"2. median RecoveredLift of `{g['primary_method']}` ≥ **{g['min_median_recovered_lift']}**;",
              f"3. `{g['primary_method']}` beats the strongest non-learned baseline "
              f"({', '.join(g['non_learned_baselines'])}) with a paired-bootstrap {int(g['ci_level'] * 100)}% CI "
              f"excluding zero (resampling identical source×target×task cells)"
              + (";\n4. on gold-answer likelihood (RecoveredNLL, `uag graded-transfer`) it also beats the strongest "
                 "non-learned baseline with a CI excluding zero." if g.get("graded_beats_baselines") else "."), "",
              f"Lift is measured against the **{g.get('baseline', 'zeroshot')}** base"
              + (f" ({decl['fewshot']['k']} worked training examples in the prompt, shot seed "
                 f"{decl['fewshot']['seed']})" if g.get("baseline") == "fewshot" else "")
              + f". Cells whose mean direct-LoRA lift is below {g['min_direct_lift']} are ineligible "
                f"(ratio unstable).", ""]

    if gate is not None:
        verdict = "**GO** — proceed to Stage 1" if gate["passed"] else "**NO-GO** — see decision tree (spec §19)"
        L += ["## Verdict", "", verdict, "", "| Criterion | Value | Threshold | Passed |", "|---|---|---|---|"]
        c = gate["criteria"]
        L.append(f"| Heterogeneous pairs passing | {c['heterogeneous_pairs']['value']} | "
                 f"≥ {c['heterogeneous_pairs']['threshold']} | {c['heterogeneous_pairs']['passed']} |")
        L.append(f"| Median RecoveredLift | {_fmt(c['median_recovered_lift']['value'])} | "
                 f"≥ {c['median_recovered_lift']['threshold']} | {c['median_recovered_lift']['passed']} |")
        v = c["beats_strongest_baseline"]["value"]
        val = (f"Δ={_fmt(v['diff'])} [{_fmt(v['low'])}, {_fmt(v['high'])}] vs `{gate['strongest_baseline']}` (n={v['n']})"
               if v else "–")
        L.append(f"| Beats strongest baseline | {val} | CI > 0 | {c['beats_strongest_baseline']['passed']} |")
        if "graded_beats_baselines" in c:
            gc = c["graded_beats_baselines"]
            gv = gc["value"]
            gval = (f"Δ RecoveredNLL={_fmt(gv['diff'])} [{_fmt(gv['low'])}, {_fmt(gv['high'])}] vs "
                    f"`{gc['strongest_baseline']}` (n={gv['n']})" if gv else gc.get("note", "–"))
            L.append(f"| Graded (gold-answer NLL) beats strongest baseline | {gval} | CI > 0 | {gc['passed']} |")
        L += ["", "### Ordered pairs (heterogeneous only)", "", "| Source → Target | Eligible tasks | Positive | Passes |",
              "|---|---|---|---|"]
        for r in gate["pairs"]:
            L.append(f"| {r['source']} → {r['target']} | {r['n_eligible_tasks']} | {r['n_positive']} | {r['passes']} |")
        if not gate["pairs"]:
            L.append("| (no heterogeneous pairs with eligible cells) | | | |")
        L += ["", "### Baselines (mean held-out score over cells shared with the primary method)", "",
              "| Baseline | Cells | Mean score |", "|---|---|---|"]
        for b in gate["baselines"]:
            L.append(f"| {b['baseline']} | {b['n']} | {_fmt(b['mean_score'])} |")
        L.append("")
    else:
        L += ["## Verdict", "", "No transfer cells were evaluated; the gate cannot be assessed.", ""]

    L += ["## Headroom: raw base vs direct LoRA (every base × task cell)", "",
          f"Lift = direct − {exp.gate.baseline} base. Flags: **ceiling** = a base scores ≥ {CEILING} "
          f"(no room to measure transfer); **format** = the few-shot prompt alone recovers ≥ "
          f"{FORMAT_SHARE:.0%} of the zero-shot-referenced lift (the adapter mostly teaches output format).", "",
          "| Base | Task | Metric | Zero-shot | Few-shot | Direct (mean ± sd over seeds) | Lift | Eligible | Flags |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in direct_rows:
        flags = ", ".join(f for f, on in (("ceiling", r["ceiling"]), ("format", r["format_dominated"])) if on)
        L.append(f"| {r['base']} | {r['task']} | {r['metric']} | {_fmt(r['s_base_zeroshot'])} | "
                 f"{_fmt(r['s_base_fewshot'])} | "
                 f"{_fmt(r['s_direct_mean'])} ± {_fmt(r['s_direct_std'])} (n={r['n_seeds']}) | "
                 f"{_fmt(r['direct_lift'])} | {r['eligible']} | {flags or '–'} |")
    L.append("")

    if cells:
        L += ["## Held-out transfer cells (seed-averaged)", "",
              f"RecoveredLift = (S_transfer − S_base) / (S_direct − S_base) with S_base the "
              f"{exp.gate.baseline} base; '–' marks ineligible cells.", "",
              "| Source → Target | Split | Task | Method | S_transfer | RecoveredLift (mean ± sd) | Δ rel. err | Seeds |",
              "|---|---|---|---|---|---|---|---|"]
        grouped = defaultdict(list)
        for c in cells:
            grouped[(c["source"], c["target"], c["split"], c["task"], c["method"])].append(c)
        for (s, t, sp, task, m), cs in sorted(grouped.items()):
            rl = seed_summary([c["recovered_lift"] for c in cs if c["recovered_lift"] is not None])
            st = float(np.mean([c["s_transfer"] for c in cs]))
            errs = [recon[(f"{s}__to__{t}__{sp}__{m}", c["seed"], task)]["mean_rel_err"]
                    for c in cs if (f"{s}__to__{t}__{sp}__{m}", c["seed"], task) in recon
                    and "mean_rel_err" in recon[(f"{s}__to__{t}__{sp}__{m}", c["seed"], task)]]
            L.append(f"| {s} → {t} | {sp} | {task} | {m} | {_fmt(st)} | "
                     f"{_fmt(rl['mean'])} ± {_fmt(rl['std'])} | {_fmt(float(np.mean(errs)) if errs else None)} | "
                     f"{len(cs)} |")
        L.append("")

    controls = [s for s in summaries if s.get("eval_task", s["task_id"]) != s["task_id"]]
    if controls:
        base_scores = {(s["target_base"], s["task_id"]): s["primary"] for s in summaries if s["role"] == "base"}
        L += ["## Collateral: control-task deltas", "",
              "Adapters are evaluated zero-shot on control tasks, so the reference here is the zero-shot base.", "",
              "| Kind | Target | Adapter task | Control task | Δ vs base |",
              "|---|---|---|---|---|"]
        acc = defaultdict(list)
        for s in controls:
            b = base_scores.get((s["target_base"], s["eval_task"]))
            if b is not None:
                acc[(s["role"] + ("/" + s["method"] if s.get("method") else ""), s["target_base"], s["task_id"],
                     s["eval_task"])].append(s["primary"] - b)
        for k, v in sorted(acc.items()):
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {_fmt(float(np.mean(v)))} |")
        L.append("")

    alt = [s for s in summaries if s.get("template") in exp.alt_templates and s["role"] == "transfer"]
    if alt:
        _, _, alt_cells = build_tables(exp, summaries, template=exp.alt_templates[0])
        rl = [c["recovered_lift"] for c in alt_cells if c["recovered_lift"] is not None
              and c["method"] == exp.gate.primary_method]
        L += ["## Robustness: alternate prompt format", "",
              f"`{exp.alt_templates[0]}`: median RecoveredLift of `{exp.gate.primary_method}` = "
              f"{_fmt(float(np.median(rl)) if rl else None)} over {len(rl)} eligible cells.", ""]

    excl = read_exclusions(paths)
    L += ["## Exclusion / failure log", ""]
    if excl:
        L += ["| Kind | Reason | Detail |", "|---|---|---|"]
        for e in excl:
            detail = e.get("run_id") or e.get("map_id") or f"{e.get('source')}→{e.get('target')}"
            L.append(f"| {e['kind']} | {e['reason']} | {detail} |")
    else:
        L.append("No exclusions recorded.")
    L += ["", "## Reproduce", "", "```bash", f"uag analyze --experiment <config for {exp.name}>", "```", ""]
    return "\n".join(L)
