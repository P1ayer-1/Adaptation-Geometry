"""Side-split projections and paired tests for the crossed task x initialisation experiment.

Follow-up to crossed_eval.py (review item: a two-sided projection cannot say which side causes
the loss). For each held-out task we fit rank-k bases on the other nine tasks and project the
held-out adapter on one side only:
  input-only   dW' = B A V Vᵀ      (keeps the adapter's own output directions)
  output-only  dW' = U Uᵀ B A      (keeps the adapter's own input directions)
  both         dW' = U Uᵀ B A V Vᵀ (as in crossed_eval.py)
and report the captured share of ||dW||_F^2 (all 168 modules) and, for the four scored tasks,
the task score of the projected adapter. Finally, paired comparisons (exact McNemar on discordant
pairs and a paired bootstrap on the accuracy difference) of every paraphrase variant against the
base model, using the per-example files written by the evaluations.

usage: python scripts/crossed_eval_sides.py [--no-eval] [--paired-only] [--k 16]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy import stats

from uag.alignment import strip_peft_prefix
from uag.config import load_base, load_experiment
from uag.evaluate import run_evaluation
from uag.extract_delta import DeltaSet, from_factors

ap = argparse.ArgumentParser()
ap.add_argument("--no-eval", action="store_true")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--paired-only", action="store_true", help="only redo the paired tests, updating the existing JSON")
args = ap.parse_args()
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent.parent
BASE = "dev_qwen2.5-0.5b"
TASKS = ["T1_hidden_rule", "T2_nli", "T3_paraphrase", "T4_json", "T5_concise", "T6_verbose",
         "T7_python", "T8_arithmetic", "T9_clinical", "T10_format"]
EVAL_TASKS = ["T3_paraphrase", "T4_json", "T8_arithmetic", "T9_clinical"]
OUT = ROOT / "results" / "crossed_eval"
PRED = ROOT / "artifacts" / "crossed_eval_sides"
SIDES = ("input_only", "output_only", "both")


def run_dir(exp, task, seed):
    return ROOT / "artifacts" / exp / "runs" / f"{BASE}_{task}_seed{seed}_r16"


def load_ab(exp, task, seed):
    d = run_dir(exp, task, seed) / "adapter"
    cfg = json.loads((d / "adapter_config.json").read_text())
    w = load_file(str(d / "adapter_model.safetensors"))
    out = {}
    for ka in sorted(k for k in w if ".lora_A." in k):
        name = strip_peft_prefix(ka.split(".lora_A.")[0])
        out[name] = (w[ka].double(), w[ka.replace(".lora_A.", ".lora_B.")].double())
    return out, cfg["lora_alpha"] / cfg["r"]


if args.paired_only:
    res = json.load(open(OUT / "crossed_eval_sides.json"))

groups = {} if args.paired_only else {
    "shared_init0": {t: load_ab("dev_3080_crossed_shared", t, 0) for t in TASKS},
    "shared_init1": {t: load_ab("dev_3080_crossed_shared", t, 1) for t in TASKS},
    "indep_seed0": {t: load_ab("dev_a100_v2", t, 0) for t in TASKS},
}
if not args.paired_only:
    modules = sorted(groups["shared_init0"]["T3_paraphrase"][0])
    res = {"base": BASE, "tasks": TASKS, "k": args.k, "n_modules": len(modules), "sides": SIDES}
_bases = {}


def bases(src, task, m, k):
    key = (src, task, m)
    if key not in _bases:
        g = groups[src]
        others = [x for x in TASKS if x != task]
        A = torch.cat([g[t][0][m][0] for t in others])
        B = torch.cat([g[t][0][m][1] for t in others], 1)
        V = torch.linalg.svd(A, full_matrices=False)[2][:k].T.clone()  # clone: a view would keep the full SVD alive
        U = torch.linalg.svd(B, full_matrices=False)[0][:, :k].clone()
        _bases[key] = (U, V)
    return _bases[key]


def project(A, B, U, V, side):
    """Return (B', A') with B' A' = projected update."""
    if side == "input_only":
        return B, (A @ V) @ V.T
    if side == "output_only":
        return U @ (U.T @ B), A
    return U @ (U.T @ B), (A @ V) @ V.T


# ---------------------------------------------------------------- captured energy
conds = [("shared_init0", "shared_init0", "same init, other tasks"),
         ("shared_init0", "shared_init1", "other init, other tasks"),
         ("shared_init0", "indep_seed0", "independent inits, other tasks"),
         ("indep_seed0", "indep_seed0", "independent inits, other tasks")]
cap = []
for held, src, desc in ([] if args.paired_only else conds):
    for t in TASKS:
        ad = groups[held][t][0]
        tot = {s: 0.0 for s in SIDES}
        den = 0.0
        for m in modules:
            A, B = ad[m]
            U, V = bases(src, t, m, args.k)
            den += float((B @ A).pow(2).sum())
            for s in SIDES:
                Bp, Ap = project(A, B, U, V, s)
                tot[s] += float((Bp @ Ap).pow(2).sum())
        cap.append(dict(heldout_group=held, basis_group=src, desc=desc, task=t,
                        **{s: tot[s] / den for s in SIDES}))
    for s in SIDES:
        vals = [x[s] for x in cap if x["heldout_group"] == held and x["basis_group"] == src]
        print(f"capture {held} <- {src} {s:12s}: median {np.median(vals):.3f} range {min(vals):.3f}-{max(vals):.3f}", flush=True)
if not args.paired_only:
    res["capture_by_side"] = cap
json.dump(res, open(OUT / "crossed_eval_sides.json", "w"), indent=1)

# ---------------------------------------------------------------- scores
if not (args.no_eval or args.paired_only):
    exp = load_experiment(ROOT / "configs/experiments/dev_3080_crossed_shared.yaml", allow_unpinned=True)
    base = load_base(ROOT / "configs/bases/dev_qwen2.5-0.5b.yaml")
    tasks = {t.task_id: t for t in exp.tasks}
    cache, scores = {}, []
    for t in EVAL_TASKS:
        for held, src in (("shared_init0", "shared_init0"), ("shared_init0", "shared_init1"),
                          ("indep_seed0", "indep_seed0")):
            ad, scale = groups[held][t]
            for side in ("input_only", "output_only"):
                mods = {}
                for m in modules:
                    A, B = ad[m]
                    U, V = bases(src, t, m, args.k)
                    Bp, Ap = project(A, B, U, V, side)
                    mods[m] = from_factors(Bp.numpy(), Ap.numpy(), scale)
                d = PRED / f"{held}__{src}__{t}__{side}"
                DeltaSet({"task_id": t, "kind": "projected", "side": side, "heldout_group": held,
                          "basis_group": src, "k": args.k}, mods).save(d)
                eid = f"crossed_{t}_projected_{side}_{held}_from_{src}"
                s = run_evaluation(base, tasks[t], OUT / "raw", eid, split="test", data_dir=exp.path("data"),
                                   max_examples=200, device="auto", dtype="bfloat16", model_cache=cache, delta_dir=d)
                scores.append(dict(task=t, variant=f"projected {side} {held}<-{src}", score=s["primary"], n=s["n"]))
                print(f"score {t} {side} {held}<-{src}: {s['primary']:.3f}", flush=True)
                res["scores_by_side"] = scores
                json.dump(res, open(OUT / "crossed_eval_sides.json", "w"), indent=1)

# ---------------------------------------------------------------- paired tests (paraphrase)
def load_scores(p):
    rows = [json.loads(l) for l in open(p)]
    return {r["example_id"]: (r["score"], r["prediction"]) for r in rows}


raw = OUT / "raw"
variants = {"base": ROOT / "results/raw/dev_a100_v2" / f"base__{BASE}__T3_paraphrase__v1.jsonl"}
for p in sorted(raw.glob("crossed_T3_paraphrase_*.jsonl")):
    variants[p.stem.replace("crossed_T3_paraphrase_", "")] = p
have = {k: load_scores(p) for k, p in variants.items() if p.exists()}
ids = sorted(have["base"])
paired = {}
for k, v in have.items():
    if k == "base" or sorted(v) != ids:
        continue
    rng = np.random.default_rng(0)  # per variant, so an interval does not depend on file order
    a = np.array([v[i][0] for i in ids]); b = np.array([have["base"][i][0] for i in ids])
    d = a - b
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    paired[k] = dict(score=float(a.mean()), base=float(b.mean()), diff=float(d.mean()),
                     ci95_paired_bootstrap=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                     discordant_variant_only=n10, discordant_base_only=n01,
                     mcnemar_exact_p=float(stats.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0,
                     prediction_agreement_with_base=float(np.mean([v[i][1] == have["base"][i][1] for i in ids])), n=len(ids))
    print(f"paired vs base {k}: {paired[k]}", flush=True)
res["paired_vs_base_T3_paraphrase"] = paired
json.dump(res, open(OUT / "crossed_eval_sides.json", "w"), indent=1)
print("crossed_eval_sides: done")
