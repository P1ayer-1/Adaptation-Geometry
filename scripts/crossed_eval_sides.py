"""Side-split projections and paired tests for the crossed task x initialisation experiment.

Follow-up to crossed_eval.py. For each held-out task we fit rank-k bases on the other nine tasks
and project the held-out adapter on one side or both:
  input_only   dW' = B A V Vᵀ      (keeps the adapter's own output directions)
  output_only  dW' = U Uᵀ B A      (keeps the adapter's own input directions)
  both         dW' = U Uᵀ B A V Vᵀ (as in crossed_eval.py)
Three ways to fit the bases:
  centred_factor    paper-style PCA: top principal directions of the stacked raw A rows (V) and
                    B columns (U) after subtracting their mean; depends on the factorisation.
  uncentred_factor  the same without centring (top singular vectors of the raw stacks).
  invariant         top eigenvectors of C_in = sum_t dW_tᵀ dW_t (V) and C_out = sum_t dW_t dW_tᵀ (U),
                    uncentred, from each update's compact SVD; independent of the factorisation.
Conditions (held-out group <- basis group): each shared initialisation held out against itself
and against the other one, and the independent-initialisation group against itself.

Outputs (results/crossed_eval/crossed_eval_sides.json):
  capture_by_side  captured share of ||dW||_F^2 (all 168 modules), every basis, condition, side, task
  scores           test scores (n = 200): projected adapters for all ten tasks with invariant bases
                   and for EVAL_TASKS with centred factor bases; direct adapters of every group; the
                   base model. All evaluated in this run's environment.
  paired_vs_base   exact McNemar on discordant pairs and a paired bootstrap on the score difference,
                   every scored variant against the base model scored in the same environment.

Caching: every evaluation is named by a fingerprint of its full configuration (kind, task, basis,
rank, side, groups, SHA-256 of every adapter file involved, evaluation settings, projection code
version) and stored with a <eval_id>.config.json sidecar. A stored result is reused only when the
sidecar matches exactly, so changing --k or any input can never mix old scores into a new run.

usage: python scripts/crossed_eval_sides.py [--no-eval] [--paired-only] [--k 16]
"""
import argparse
import hashlib
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
EVAL_TASKS = ["T3_paraphrase", "T4_json", "T8_arithmetic", "T9_clinical"]  # factor-basis scores
OUT = ROOT / "results" / "crossed_eval"
RAW = OUT / "raw_sides"
PRED = ROOT / "artifacts" / "crossed_eval_sides"
SIDES = ("input_only", "output_only", "both")
BASES = ("centred_factor", "uncentred_factor", "invariant")
SCORED_BASES = ("centred_factor", "invariant")  # uncentred_factor: captured energy only
CONDS = [("shared_init0", "shared_init0", "same init"), ("shared_init0", "shared_init1", "other init"),
         ("shared_init1", "shared_init1", "same init"), ("shared_init1", "shared_init0", "other init"),
         ("indep_seed0", "indep_seed0", "independent inits")]
GROUP_SRC = {"shared_init0": ("dev_3080_crossed_shared", 0), "shared_init1": ("dev_3080_crossed_shared", 1),
             "indep_seed0": ("dev_a100_v2", 0)}
PROJECTION_VERSION = 2  # bump whenever basis fitting or projection changes
EVAL_KW = dict(split="test", max_examples=200, device="auto", dtype="bfloat16", batch_size=16)
JSON_OUT = OUT / "crossed_eval_sides.json"


def run_dir(group, task):
    exp, seed = GROUP_SRC[group]
    return ROOT / "artifacts" / exp / "runs" / f"{BASE}_{task}_seed{seed}_r16"


@lru_cache(maxsize=None)
def adapter_sha(group, task):
    h = hashlib.sha256()
    with open(run_dir(group, task) / "adapter" / "adapter_model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_ab(group, task):
    d = run_dir(group, task) / "adapter"
    cfg = json.loads((d / "adapter_config.json").read_text())
    w = load_file(str(d / "adapter_model.safetensors"))
    out = {}
    for ka in sorted(k for k in w if ".lora_A." in k):
        name = strip_peft_prefix(ka.split(".lora_A.")[0])
        out[name] = (w[ka].double(), w[ka.replace(".lora_A.", ".lora_B.")].double())
    return out, cfg["lora_alpha"] / cfg["r"]


def factors(A, B):
    """Compact SVD of each B_t A_t via QR: U (n,dout,r), S (n,r), V (n,din,r)."""
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.transpose(1, 2))
    u, s, vh = torch.linalg.svd(Rb @ Ra.transpose(1, 2))
    return Qb @ u, s, Qa @ vh.transpose(1, 2)


def top_right(rows, k, centre=False):
    if centre:
        rows = rows - rows.mean(0)
    return torch.linalg.svd(rows, full_matrices=False)[2][:k].T.clone()  # clone frees the full SVD


def project(A, B, U, V, side):
    """Return (B', A') with B' A' = projected update."""
    if side == "input_only":
        return B, (A @ V) @ V.T
    if side == "output_only":
        return U @ (U.T @ B), A
    return U @ (U.T @ B), (A @ V) @ V.T


def others(task):
    return [x for x in TASKS if x != task]


if args.paired_only:
    res = json.load(open(JSON_OUT))
else:
    RAW.mkdir(parents=True, exist_ok=True)
    groups = {g: {t: load_ab(g, t) for t in TASKS} for g in GROUP_SRC}
    modules = sorted(groups["shared_init0"]["T3_paraphrase"][0])
    res = {"base": BASE, "tasks": TASKS, "eval_tasks_factor": EVAL_TASKS, "k": args.k, "n_modules": len(modules),
           "sides": SIDES, "bases": BASES, "scored_bases": SCORED_BASES, "conditions": CONDS, "projection_version": PROJECTION_VERSION,
           "adapter_sha256": {g: {t: adapter_sha(g, t) for t in TASKS} for g in GROUP_SRC}}

    @lru_cache(maxsize=2)
    def bases_for(src, task, basis):
        """{module: (U d_out x k, V d_in x k)} fitted on the nine tasks other than `task`."""
        g = groups[src]
        out = {}
        for m in modules:
            A = torch.stack([g[t][0][m][0] for t in others(task)])  # (9, r, d_in)
            B = torch.stack([g[t][0][m][1] for t in others(task)])  # (9, d_out, r)
            if basis in ("centred_factor", "uncentred_factor"):
                c = basis == "centred_factor"
                V = top_right(A.reshape(-1, A.shape[-1]), args.k, centre=c)
                U = top_right(B.transpose(1, 2).reshape(-1, B.shape[1]), args.k, centre=c)
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
                print(f"capture {basis:16s} {held} <- {src} {s:12s}: median {np.median(v):.3f} "
                      f"range {min(v):.3f}-{max(v):.3f}", flush=True)
    res["capture_by_side"] = cap
    json.dump(res, open(JSON_OUT, "w"), indent=1)

    # ------------------------------------------------------------ scores
    if not args.no_eval:
        exp = load_experiment(ROOT / "configs/experiments/dev_3080_crossed_shared.yaml", allow_unpinned=True)
        base = load_base(ROOT / "configs/bases/dev_qwen2.5-0.5b.yaml")
        tasks = {t.task_id: t for t in exp.tasks}
        cache, scores = {}, []
        env = dict(torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        res["eval_environment"] = env

        def score(cfg, label, build=None, **extra):
            """Evaluate (or reuse, if the stored config matches exactly) and record one variant."""
            t = cfg["task"]
            cfg = dict(cfg, eval=EVAL_KW, env=env, base=BASE, data=exp.path("data").name,
                       task_config_sha256=file_sha(ROOT / f"configs/tasks/v2/{t}.yaml"))
            fp = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
            eid = f"sides_{t}_{label.replace(' ', '_').replace('<-', '_from_')}_{fp}"
            side_f, summ_f = RAW / f"{eid}.config.json", RAW / f"{eid}.summary.json"
            if summ_f.exists() and side_f.exists() and json.load(open(side_f)) == cfg:
                s, reused = json.load(open(summ_f)), True
            else:
                if build is not None:
                    extra["delta_dir"] = build()
                s, reused = run_evaluation(base, tasks[t], RAW, eid, data_dir=exp.path("data"), model_cache=cache,
                                           **EVAL_KW, **extra), False
                json.dump(cfg, open(side_f, "w"), indent=1, sort_keys=True)
            scores.append(dict(task=t, variant=label, kind=cfg["kind"], basis=cfg.get("basis"), side=cfg.get("side"),
                               heldout_group=cfg.get("heldout_group"), basis_group=cfg.get("basis_group"),
                               eval_id=eid, score=s["primary"], n=s["n"], reused=reused))
            print(f"score {t} {label}: {s['primary']:.3f}{' (reused)' if reused else ''}", flush=True)
            res["scores"] = scores
            json.dump(res, open(JSON_OUT, "w"), indent=1)

        for t in TASKS:
            score(dict(kind="base", task=t), "base")
            for g in GROUP_SRC:
                score(dict(kind="direct", task=t, group=g, adapter_sha256=adapter_sha(g, t)), f"direct {g}",
                      adapter_dir=run_dir(g, t) / "adapter")
            for basis in SCORED_BASES:
                if basis != "invariant" and t not in EVAL_TASKS:
                    continue
                for held, src, _ in CONDS:
                    ad, scale = groups[held][t]
                    for side in SIDES:
                        cfg = dict(kind="projected", task=t, basis=basis, k=args.k, side=side, heldout_group=held,
                                   basis_group=src, projection_version=PROJECTION_VERSION,
                                   heldout_sha256=adapter_sha(held, t),
                                   basis_sha256={x: adapter_sha(src, x) for x in others(t)})

                        def build(t=t, basis=basis, held=held, src=src, side=side, ad=ad, scale=scale, cfg=cfg):
                            bs = bases_for(src, t, basis)
                            mods = {m: from_factors(*[x.numpy() for x in project(*ad[m], *bs[m], side)], scale)
                                    for m in modules}
                            d = PRED / f"{basis}_k{args.k}__{held}__{src}__{t}__{side}"
                            DeltaSet({"task_id": t, **{k: v for k, v in cfg.items() if k != "task"}}, mods).save(d)
                            return d
                        score(cfg, f"{basis} k{args.k} {side} {held}<-{src}", build=build)


# ---------------------------------------------------------------- paired tests
def load_scores(p):
    rows = [json.loads(l) for l in open(p)]
    return {r["example_id"]: (r["score"], r["prediction"]) for r in rows}


paired = {}
for t in TASKS:
    entries = [x for x in res.get("scores", []) if x["task"] == t]
    base_e = next((x for x in entries if x["kind"] == "base"), None)
    if base_e is None:
        continue
    base_scores = load_scores(RAW / f"{base_e['eval_id']}.jsonl")
    ids = sorted(base_scores)
    b = np.array([base_scores[i][0] for i in ids], dtype=float)
    paired[t] = {}
    for x in entries:
        if x["kind"] == "base":
            continue
        v = load_scores(RAW / f"{x['eval_id']}.jsonl")
        assert sorted(v) == ids, x["eval_id"]
        a = np.array([v[i][0] for i in ids], dtype=float)
        d = a - b
        n10, n01 = int((a > b).sum()), int((a < b).sum())
        rng = np.random.default_rng(0)  # per variant, so an interval does not depend on order
        boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        paired[t][x["variant"]] = dict(
            eval_id=x["eval_id"], score=float(a.mean()), base=float(b.mean()), diff=float(d.mean()),
            ci95_paired_bootstrap=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            discordant_variant_better=n10, discordant_base_better=n01,
            mcnemar_exact_p=float(stats.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0,
            prediction_agreement_with_base=float(np.mean([v[i][1] == base_scores[i][1] for i in ids])), n=len(ids))
    for k, x in paired[t].items():
        print(f"paired {t} {k}: {x['score']:.3f} vs {x['base']:.3f}, diff {x['diff']:+.3f} "
              f"CI [{x['ci95_paired_bootstrap'][0]:.3f}, {x['ci95_paired_bootstrap'][1]:.3f}] "
              f"+{x['discordant_variant_better']}/-{x['discordant_base_better']} p={x['mcnemar_exact_p']:.3g} "
              f"agree={x['prediction_agreement_with_base']:.2f}", flush=True)
res["paired_vs_base"] = paired
json.dump(res, open(JSON_OUT, "w"), indent=1)
print("crossed_eval_sides: done")
