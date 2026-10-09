"""Crossed task x initialisation test for the universal-subspace note.

Same 10 v2 tasks on Qwen2.5-0.5B under two LoRA initialisation regimes:
  shared  dev_3080_crossed_shared: every task on a seed shares one random A (seeds 0, 1 = two inits)
  indep   dev_a100_v2: each (task, seed) has its own random A
plus a hardware check (dev_3080_crossed_indep reruns T3/T8 of dev_a100_v2 on the 3080 with the
same per-task init seeds).

1. geometry  paper-style A spectrum (stack rank rows, centre, PCA) per module, per group of 10
             adapters, against independent random inits at the same N.
2. capture   leave-one-task-out: fit a rank-16 input basis (top right singular vectors of the
             other tasks' stacked A rows) and output basis (top left singular vectors of their
             stacked B columns), project the held-out adapter (dW' = U Uᵀ B A V Vᵀ) and report
             the captured share of ||dW||_F^2 over all modules. Bases from: the same shared-init
             group, the other shared-init group (different init, same tasks), the independent group.
3. scores    task score of projected held-out adapters (T3, T4, T8, T9) vs the direct adapter.

usage: python scripts/crossed_eval.py [--no-eval]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from uag.alignment import strip_peft_prefix
from uag.config import load_base, load_experiment
from uag.evaluate import run_evaluation
from uag.extract_delta import DeltaSet, from_factors

ap = argparse.ArgumentParser()
ap.add_argument("--no-eval", action="store_true")
ap.add_argument("--k", type=int, default=16)
args = ap.parse_args()
ROOT = Path(__file__).resolve().parent.parent
BASE = "dev_qwen2.5-0.5b"
TASKS = ["T1_hidden_rule", "T2_nli", "T3_paraphrase", "T4_json", "T5_concise", "T6_verbose",
         "T7_python", "T8_arithmetic", "T9_clinical", "T10_format"]
EVAL_TASKS = ["T3_paraphrase", "T4_json", "T8_arithmetic", "T9_clinical"]
OUT = ROOT / "results" / "crossed_eval"
OUT.mkdir(parents=True, exist_ok=True)
PRED = ROOT / "artifacts" / "crossed_eval"
gen = torch.Generator().manual_seed(0)


def run_dir(exp, task, seed):
    return ROOT / "artifacts" / exp / "runs" / f"{BASE}_{task}_seed{seed}_r16"


def load_ab(exp, task, seed):
    """{module: (A r x d_in, B d_out x r)} in float64, plus the LoRA scale."""
    d = run_dir(exp, task, seed) / "adapter"
    cfg = json.loads((d / "adapter_config.json").read_text())
    w = load_file(str(d / "adapter_model.safetensors"))
    out = {}
    for ka in sorted(k for k in w if ".lora_A." in k):
        name = strip_peft_prefix(ka.split(".lora_A.")[0])
        out[name] = (w[ka].double(), w[ka.replace(".lora_A.", ".lora_B.")].double())
    return out, cfg["lora_alpha"] / cfg["r"]


groups = {
    "shared_init0": {t: load_ab("dev_3080_crossed_shared", t, 0) for t in TASKS},
    "shared_init1": {t: load_ab("dev_3080_crossed_shared", t, 1) for t in TASKS},
    "indep_seed0": {t: load_ab("dev_a100_v2", t, 0) for t in TASKS},
    "indep_seed1": {t: load_ab("dev_a100_v2", t, 1) for t in TASKS},
}
modules = sorted(groups["shared_init0"]["T3_paraphrase"][0])
res = {"base": BASE, "tasks": TASKS, "k": args.k, "n_modules": len(modules)}


def top16_share(rows):
    rows = rows - rows.mean(0)
    G = rows @ rows.T
    ev = torch.linalg.eigvalsh(G).clamp(min=0).flip(0)
    return float(ev[:16].sum() / ev.sum())


# ---------------------------------------------------------------- 0. hardware check + sanity
hw = {}
for t in ("T3_paraphrase", "T8_arithmetic"):
    for s in (0, 1):
        a, _ = load_ab("dev_3080_crossed_indep", t, s)
        b, _ = groups[f"indep_seed{s}"][t]
        num = sum(float(((B1 @ A1) * (B2 @ A2)).sum()) for (A1, B1), (A2, B2) in zip(a.values(), b.values()))
        den = np.sqrt(sum(float((B1 @ A1).pow(2).sum()) for A1, B1 in a.values()) *
                      sum(float((B2 @ A2).pow(2).sum()) for A2, B2 in b.values()))
        a0 = float(torch.stack([torch.nn.functional.cosine_similarity(a[m][0].flatten(), b[m][0].flatten(), 0)
                                for m in modules]).mean())
        hw[f"{t} seed{s}"] = dict(dW_cosine_3080_vs_A100=num / den, A_cosine=a0)
res["hardware_check"] = hw
init_cos = {}
for nm, g in groups.items():
    cs = []
    for i, t1 in enumerate(TASKS):
        for t2 in TASKS[i + 1:]:
            cs.append(float(torch.nn.functional.cosine_similarity(
                g[t1][0][modules[0]][0].flatten(), g[t2][0][modules[0]][0].flatten(), 0)))
    init_cos[nm] = dict(mean=float(np.mean(cs)), min=float(np.min(cs)), max=float(np.max(cs)))
res["A_pairwise_cosine_first_module"] = init_cos
print(json.dumps({"hardware_check": hw, "A_pairwise_cosine": init_cos}, indent=1), flush=True)

# ---------------------------------------------------------------- 1. geometry
geo = {}
for nm, g in groups.items():
    vals = [top16_share(torch.cat([g[t][0][m][0] for t in TASKS])) for m in modules]
    geo[nm] = dict(median=float(np.median(vals)), p10=float(np.percentile(vals, 10)), p90=float(np.percentile(vals, 90)))
both = [top16_share(torch.cat([groups[g][t][0][m][0] for g in ("shared_init0", "shared_init1") for t in TASKS]))
        for m in modules]
geo["shared_both_inits"] = dict(median=float(np.median(both)), p10=float(np.percentile(both, 10)),
                                p90=float(np.percentile(both, 90)))
nul = []
for m in modules:
    A = groups["indep_seed0"]["T3_paraphrase"][0][m][0]
    b = 1 / np.sqrt(A.shape[1])
    nul.append(top16_share((torch.rand((len(TASKS) * A.shape[0], A.shape[1]), generator=gen, dtype=torch.float64) * 2 - 1) * b))
geo["independent_random_inits"] = dict(median=float(np.median(nul)), p10=float(np.percentile(nul, 10)),
                                       p90=float(np.percentile(nul, 90)))
res["geometry_top16_A"] = geo
print(json.dumps(geo, indent=1), flush=True)


# ---------------------------------------------------------------- 2. capture
def bases(g, tasks, m, k):
    A = torch.cat([g[t][0][m][0] for t in tasks])            # (n r) x d_in
    B = torch.cat([g[t][0][m][1] for t in tasks], 1)         # d_out x (n r)
    V = torch.linalg.svd(A, full_matrices=False)[2][:k].T    # d_in x k
    U = torch.linalg.svd(B, full_matrices=False)[0][:, :k]   # d_out x k
    return U, V


def captured(adapter, basis_group, basis_tasks, k):
    tot = cap = 0.0
    for m in modules:
        A, B = adapter[m]
        U, V = bases(basis_group, basis_tasks, m, k)
        dW = B @ A
        tot += float(dW.pow(2).sum())
        cap += float((U.T @ B @ A @ V).pow(2).sum())
    return cap / tot


conds = [("shared_init0", "shared_init0", "same init, other tasks"),
         ("shared_init0", "shared_init1", "other init, other tasks"),
         ("shared_init0", "indep_seed0", "independent inits, other tasks"),
         ("indep_seed0", "indep_seed0", "independent inits, other tasks")]
cap = []
for held, src, desc in conds:
    for t in TASKS:
        others = [x for x in TASKS if x != t]
        c = captured(groups[held][t][0], groups[src], others, args.k)
        cap.append(dict(heldout_group=held, basis_group=src, desc=desc, task=t, captured=c))
    vals = [x["captured"] for x in cap if x["heldout_group"] == held and x["basis_group"] == src]
    print(f"capture {held} <- {src}: median {np.median(vals):.3f} range {min(vals):.3f}-{max(vals):.3f}", flush=True)
# same task, other init: basis from the other init's adapters including the same task
for t in TASKS:
    c = captured(groups["shared_init0"][t][0], groups["shared_init1"], TASKS, args.k)
    cap.append(dict(heldout_group="shared_init0", basis_group="shared_init1", desc="other init, all tasks incl. same task",
                    task=t, captured=c))
res["capture"] = cap
json.dump(res, open(OUT / "crossed_eval.json", "w"), indent=1)

# ---------------------------------------------------------------- 3. scores of projected adapters
if not args.no_eval:
    exp = load_experiment(ROOT / "configs/experiments/dev_3080_crossed_shared.yaml", allow_unpinned=True)
    base = load_base(ROOT / "configs/bases/dev_qwen2.5-0.5b.yaml")
    tasks = {t.task_id: t for t in exp.tasks}
    cache, scores = {}, []
    for t in EVAL_TASKS:
        others = [x for x in TASKS if x != t]
        variants = {"direct shared_init0": run_dir("dev_3080_crossed_shared", t, 0) / "adapter",
                    "direct indep_seed0": run_dir("dev_a100_v2", t, 0) / "adapter"}
        for held, src in (("shared_init0", "shared_init0"), ("shared_init0", "shared_init1"),
                          ("indep_seed0", "indep_seed0")):
            ad, scale = groups[held][t]
            mods = {}
            for m in modules:
                A, B = ad[m]
                U, V = bases(groups[src], others, m, args.k)
                mods[m] = from_factors((U @ (U.T @ B)).numpy(), ((A @ V) @ V.T).numpy(), scale)
            d = PRED / f"{held}__{src}__{t}"
            DeltaSet({"task_id": t, "kind": "projected", "heldout_group": held, "basis_group": src, "k": args.k},
                     mods).save(d)
            variants[f"projected {held}<-{src}"] = d
        for v, d in variants.items():
            eid = f"crossed_{t}_{v.replace(' ', '_').replace('<-', '_from_')}"
            kw = {"adapter_dir": d} if v.startswith("direct") else {"delta_dir": d}
            s = run_evaluation(base, tasks[t], OUT / "raw", eid, split="test", data_dir=exp.path("data"),
                               max_examples=200, device="auto", dtype="bfloat16", model_cache=cache, **kw)
            scores.append(dict(task=t, variant=v, score=s["primary"], n=s["n"]))
            print(f"score {t} {v}: {s['primary']:.3f}", flush=True)
            res["scores"] = scores
            json.dump(res, open(OUT / "crossed_eval.json", "w"), indent=1)
print("crossed_eval: done")
