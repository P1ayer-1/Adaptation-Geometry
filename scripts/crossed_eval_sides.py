"""Side-split projections and paired tests for the crossed task x initialisation experiment.

Follow-up to crossed_eval.py. For each held-out task we fit rank-k bases on the other nine tasks
and project the held-out adapter on one side or both:
  input_only   dW' = B A V Vᵀ      (keeps the adapter's own output directions)
  output_only  dW' = U Uᵀ B A      (keeps the adapter's own input directions)
  both         dW' = U Uᵀ B A V Vᵀ (as in crossed_eval.py)
Two ways to fit the bases:
  factor     paper-style: top singular vectors of the stacked raw A rows (V) and B columns (U);
             depends on how each update is split into B and A.
  invariant  top eigenvectors of C_in = sum_t dW_tᵀ dW_t (V) and C_out = sum_t dW_t dW_tᵀ (U),
             computed from each update's compact SVD; independent of the factorisation.
Conditions (held-out group <- basis group): each shared initialisation held out against itself
and against the other one, and the independent-initialisation group against itself.
Outputs: captured share of ||dW||_F^2 (all 168 modules) for every condition, basis and side; task
scores of the projected adapters for the four scored tasks (evaluations already on disk are
reused); direct shared_init1 scores; and paired comparisons with the base model (exact McNemar
on discordant pairs, paired bootstrap on the accuracy difference) for every scored variant.

usage: python scripts/crossed_eval_sides.py [--no-eval] [--paired-only] [--k 16]
"""
import argparse
import json
from functools import lru_cache
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
ap.add_argument("--paired-only", action="store_true", help="only redo the paired tests, updating the existing JSON")
ap.add_argument("--k", type=int, default=16)
args = ap.parse_args()
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent.parent
BASE = "dev_qwen2.5-0.5b"
TASKS = ["T1_hidden_rule", "T2_nli", "T3_paraphrase", "T4_json", "T5_concise", "T6_verbose",
         "T7_python", "T8_arithmetic", "T9_clinical", "T10_format"]
EVAL_TASKS = ["T3_paraphrase", "T4_json", "T8_arithmetic", "T9_clinical"]
OUT = ROOT / "results" / "crossed_eval"
RAW = OUT / "raw"
PRED = ROOT / "artifacts" / "crossed_eval_sides"
SIDES = ("input_only", "output_only", "both")
BASES = ("factor", "invariant")
CONDS = [("shared_init0", "shared_init0", "same init"), ("shared_init0", "shared_init1", "other init"),
         ("shared_init1", "shared_init1", "same init"), ("shared_init1", "shared_init0", "other init"),
         ("indep_seed0", "indep_seed0", "independent inits")]
JSON_OUT = OUT / "crossed_eval_sides.json"


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


GROUP_SRC = {"shared_init0": ("dev_3080_crossed_shared", 0), "shared_init1": ("dev_3080_crossed_shared", 1),
             "indep_seed0": ("dev_a100_v2", 0)}


def eval_id(t, held, src, basis, side):
    if basis == "factor":  # names used by crossed_eval.py (both) and the first side-split run
        return f"crossed_{t}_projected_{held}_from_{src}" if side == "both" else \
               f"crossed_{t}_projected_{side}_{held}_from_{src}"
    return f"crossed_{t}_projinv_{side}_{held}_from_{src}"


def factors(A, B):
    """Compact SVD of each B_t A_t via QR: U (n,dout,r), S (n,r), V (n,din,r)."""
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.transpose(1, 2))
    u, s, vh = torch.linalg.svd(Rb @ Ra.transpose(1, 2))
    return Qb @ u, s, Qa @ vh.transpose(1, 2)


def top_right(rows, k):
    return torch.linalg.svd(rows, full_matrices=False)[2][:k].T.clone()  # clone frees the full SVD


def project(A, B, U, V, side):
    """Return (B', A') with B' A' = projected update."""
    if side == "input_only":
        return B, (A @ V) @ V.T
    if side == "output_only":
        return U @ (U.T @ B), A
    return U @ (U.T @ B), (A @ V) @ V.T


if args.paired_only:
    res = json.load(open(JSON_OUT))
else:
    groups = {g: {t: load_ab(e, t, s) for t in TASKS} for g, (e, s) in GROUP_SRC.items()}
    modules = sorted(groups["shared_init0"]["T3_paraphrase"][0])
    res = {"base": BASE, "tasks": TASKS, "k": args.k, "n_modules": len(modules), "sides": SIDES,
           "bases": BASES, "conditions": CONDS}

    @lru_cache(maxsize=2)
    def bases_for(src, task, basis):
        """{module: (U d_out x k, V d_in x k)} fitted on the nine tasks other than `task`."""
        g = groups[src]
        others = [x for x in TASKS if x != task]
        out = {}
        for m in modules:
            A = torch.stack([g[t][0][m][0] for t in others])  # (9, r, d_in)
            B = torch.stack([g[t][0][m][1] for t in others])  # (9, d_out, r)
            if basis == "factor":
                V = top_right(A.reshape(-1, A.shape[-1]), args.k)
                U = top_right(B.transpose(1, 2).reshape(-1, B.shape[1]), args.k)
            else:
                Uf, S, Vf = factors(A, B)
                V = top_right((Vf * S[:, None, :]).transpose(1, 2).reshape(-1, Vf.shape[1]), args.k)
                U = top_right((Uf * S[:, None, :]).transpose(1, 2).reshape(-1, Uf.shape[1]), args.k)
            out[m] = (U, V)
        return out

    # ------------------------------------------------------------ captured energy
    cap = []
    for basis in BASES:
        for held, src, desc in CONDS:
            for t in TASKS:
                ad, bs = groups[held][t][0], bases_for(src, t, basis)
                tot, den = {s: 0.0 for s in SIDES}, 0.0
                for m in modules:
                    A, B = ad[m]
                    U, V = bs[m]
                    den += float((B @ A).pow(2).sum())
                    for s in SIDES:
                        Bp, Ap = project(A, B, U, V, s)
                        tot[s] += float((Bp @ Ap).pow(2).sum())
                cap.append(dict(basis=basis, heldout_group=held, basis_group=src, desc=desc, task=t,
                                **{s: tot[s] / den for s in SIDES}))
            for s in SIDES:
                v = [x[s] for x in cap if (x["basis"], x["heldout_group"], x["basis_group"]) == (basis, held, src)]
                print(f"capture {basis:9s} {held} <- {src} {s:12s}: median {np.median(v):.3f} "
                      f"range {min(v):.3f}-{max(v):.3f}", flush=True)
    res["capture_by_side"] = cap
    json.dump(res, open(JSON_OUT, "w"), indent=1)

    # ------------------------------------------------------------ scores
    if not args.no_eval:
        exp = load_experiment(ROOT / "configs/experiments/dev_3080_crossed_shared.yaml", allow_unpinned=True)
        base = load_base(ROOT / "configs/bases/dev_qwen2.5-0.5b.yaml")
        tasks = {t.task_id: t for t in exp.tasks}
        cache, scores = {}, []
        kw = dict(split="test", data_dir=exp.path("data"), max_examples=200, device="auto", dtype="bfloat16",
                  model_cache=cache)

        def score(t, eid, variant, **extra):
            f = RAW / f"{eid}.summary.json"
            if f.exists() and not extra.pop("force", False):
                s, reused = json.load(open(f)), True
            else:
                extra.pop("force", None)
                s, reused = run_evaluation(base, tasks[t], RAW, eid, **kw, **extra), False
            scores.append(dict(task=t, variant=variant, eval_id=eid, score=s["primary"], n=s["n"], reused=reused))
            print(f"score {t} {variant}: {s['primary']:.3f}{' (reused)' if reused else ''}", flush=True)
            res["scores_by_side"] = scores
            json.dump(res, open(JSON_OUT, "w"), indent=1)

        for t in EVAL_TASKS:
            score(t, f"crossed_{t}_direct_shared_init1", "direct shared_init1",
                  adapter_dir=run_dir("dev_3080_crossed_shared", t, 1) / "adapter")
            for basis in BASES:
                for held, src, _ in CONDS:
                    ad, scale = groups[held][t]
                    for side in SIDES:
                        eid = eval_id(t, held, src, basis, side)
                        if not (RAW / f"{eid}.summary.json").exists():
                            bs = bases_for(src, t, basis)
                            mods = {}
                            for m in modules:
                                Bp, Ap = project(*ad[m], *bs[m], side)
                                mods[m] = from_factors(Bp.numpy(), Ap.numpy(), scale)
                            d = PRED / f"{basis}__{held}__{src}__{t}__{side}"
                            DeltaSet({"task_id": t, "kind": "projected", "basis": basis, "side": side,
                                      "heldout_group": held, "basis_group": src, "k": args.k}, mods).save(d)
                            score(t, eid, f"{basis} {side} {held}<-{src}", delta_dir=d)
                        else:
                            score(t, eid, f"{basis} {side} {held}<-{src}")
        # hardware check: the two-sided factor projections were first scored on an A100; rescore here
        for t in EVAL_TASKS:
            for held, src in (("shared_init0", "shared_init0"), ("shared_init0", "shared_init1")):
                ad, scale = groups[held][t]
                bs = bases_for(src, t, "factor")
                mods = {m: from_factors(*[x.numpy() for x in project(*ad[m], *bs[m], "both")], scale) for m in modules}
                d = PRED / f"hwcheck__{held}__{src}__{t}"
                DeltaSet({"task_id": t, "kind": "projected", "basis": "factor", "side": "both",
                          "heldout_group": held, "basis_group": src, "k": args.k}, mods).save(d)
                score(t, f"hwcheck_{eval_id(t, held, src, 'factor', 'both')}", f"hwcheck factor both {held}<-{src}",
                      delta_dir=d, force=True)


# ---------------------------------------------------------------- paired tests
def load_scores(p):
    rows = [json.loads(l) for l in open(p)]
    return {r["example_id"]: (r["score"], r["prediction"]) for r in rows}


paired = {}
for t in EVAL_TASKS:
    base_scores = load_scores(ROOT / "results/raw/dev_a100_v2" / f"base__{BASE}__{t}__v1.jsonl")
    ids = sorted(base_scores)
    b = np.array([base_scores[i][0] for i in ids])
    paired[t] = {}
    for p in sorted(RAW.glob(f"*crossed_{t}_*.jsonl")):
        v = load_scores(p)
        if sorted(v) != ids:
            continue
        a = np.array([v[i][0] for i in ids])
        d = a - b
        n10 = int(((a > b)).sum()); n01 = int(((a < b)).sum())
        rng = np.random.default_rng(0)  # per variant, so an interval does not depend on file order
        boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        k = p.stem.replace(f"crossed_{t}_", "")
        paired[t][k] = dict(score=float(a.mean()), base=float(b.mean()), diff=float(d.mean()),
                            ci95_paired_bootstrap=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                            discordant_variant_better=n10, discordant_base_better=n01,
                            mcnemar_exact_p=float(stats.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0,
                            prediction_agreement_with_base=float(np.mean([v[i][1] == base_scores[i][1] for i in ids])),
                            n=len(ids))
    for k, x in paired[t].items():
        print(f"paired {t} {k}: {x['score']:.3f} vs {x['base']:.3f}, diff {x['diff']:+.3f} "
              f"CI [{x['ci95_paired_bootstrap'][0]:.3f}, {x['ci95_paired_bootstrap'][1]:.3f}] "
              f"+{x['discordant_variant_better']}/-{x['discordant_base_better']} p={x['mcnemar_exact_p']:.3g} "
              f"agree={x['prediction_agreement_with_base']:.2f}", flush=True)
res["paired_vs_base"] = paired
res.pop("paired_vs_base_T3_paraphrase", None)
json.dump(res, open(JSON_OUT, "w"), indent=1)
print("crossed_eval_sides: done")
