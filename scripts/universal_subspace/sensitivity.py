"""Sensitivity and decomposition checks for the universal-subspace note (v2 review items).

usage: python sensitivity.py MANIFEST PAPER_SET.json OUT.json [--base F:/mistral] [--root F:/lol]

  all502    grouping and the full-set paper-style A spectrum on all 502 Table-10 adapters (pool, iid
            and ood) next to the 497 (pool + iid) used in the note.
  movement  distance of each group adapter's A from the group mean m: the median relative distance
            (as reported) and the root-mean-square relative distance. The RMS is a lower bound on
            the RMS distance from any common reference c (in particular the true shared
            initialisation), since sum_i ||A_i - m||^2 <= sum_i ||A_i - c||^2 for every c.
            Individual distances from m are not bounds, so the note reports the RMS as the bound.
  basefixed energy decomposition with a fixed denominator for the base-direction removal
            (singletons, both sides, k = 16 and 128). With E = total energy of the stacked rows,
            E_base = energy inside the base weight's top-k singular directions, and T16(P) = energy
            in the top-16 eigen-directions of the rows after projecting those directions out, we
            report T16/E before projection, T16(P)/E (fixed denominator) and T16(P)/(E - E_base)
            (renormalised, as in analyse.py), for the real singletons and for a strength-matched
            null (10 repeats), and the excess under each convention.
"""
import argparse
import json
import time

import numpy as np
import torch
from safetensors import safe_open

from common import (SLOTS, components, cos_matrix, factors, gram_eigs, load_factors, load_manifest,
                    paper_spectrum, quantiles, rand_frames, side_rows)

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("paper_set")
ap.add_argument("out")
ap.add_argument("--base", default="F:/mistral")
ap.add_argument("--root", default="F:/lol")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--repeats", type=int, default=10)
args = ap.parse_args()
torch.set_num_threads(12)
gen = torch.Generator().manual_seed(2)
P = json.load(open(args.paper_set))
shared_ids = set(P["grouping"]["shared_ids"])
t0 = time.time()
res = {"seed": 2, "repeats": args.repeats, "threshold": args.threshold}
def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
def save():
    json.dump(res, open(args.out, "w"), indent=1)

# ---------------------------------------------------------------- all502
ids_all, files_all = load_manifest(args.manifest, roles=("pool", "iid", "ood"), root=args.root)
ids_497, files_497 = load_manifest(args.manifest, roles=("pool", "iid"), root=args.root)
man = {a["id"]: a["role"] for a in json.load(open(args.manifest))["adapters"]}
A0, _ = load_factors(files_all, (0, "q"))
lab = components(cos_matrix(A0), args.threshold)
sizes = np.bincount(lab)
big = int(sizes.argmax())
in_group = [ids_all[i] for i in np.where(lab == big)[0]]
ood_in_group = [i for i in in_group if man[i] == "ood"]
same_497 = set(i for i in in_group if man[i] != "ood") == shared_ids
spec = {}
for slot in SLOTS:
    key = f"L{slot[0]} {slot[1]}"
    A, _ = load_factors(files_all, slot)
    s = paper_spectrum(A.reshape(-1, A.shape[-1]))
    s497 = next(x for x in P["paper"] if x["slot"] == key)["A_full"]
    spec[key] = dict(top16_502=s["top16"], k90_502=s["k90"], top16_497=s497["top16"], k90_497=s497["k90"])
    log("all502", key, round(s["top16"], 4), "vs 497", round(s497["top16"], 4))
res["all502"] = dict(n=len(ids_all), n_groups=int(len(sizes)), largest_group=int(sizes.max()),
                     singletons=int((sizes == 1).sum()), multi_member_groups=int((sizes > 1).sum()),
                     ood_adapters_in_group=ood_in_group, group_restricted_to_497_equals_note=bool(same_497),
                     A_full_spectrum=spec)
save()

# ---------------------------------------------------------------- movement (group, 497 set)
shared = np.array([i for i, x in enumerate(ids_497) if x in shared_ids])
single = np.array([i for i, x in enumerate(ids_497) if x not in shared_ids])
mov = {}
for slot in SLOTS:
    key = f"L{slot[0]} {slot[1]}"
    A, _ = load_factors([files_497[i] for i in shared], slot)
    m = A.mean(0)
    d = (A - m).flatten(1).norm(dim=1) / m.flatten().norm()
    mov[key] = dict(median=float(d.median()), rms=float(d.pow(2).mean().sqrt()), p10=float(d.quantile(0.1)),
                    p90=float(d.quantile(0.9)), n=int(len(d)))
    log("movement", key, {k: round(v, 3) for k, v in mov[key].items() if k != "n"})
res["movement_from_group_mean_rel"] = mov
save()

# ---------------------------------------------------------------- basefixed (singletons, 497 set)
idx = json.load(open(f"{args.base}/model.safetensors.index.json"))["weight_map"]
def base_w(name):
    with safe_open(f"{args.base}/" + idx[name], "pt") as h:
        return h.get_tensor(name).double()
def top16_energy(rows):
    ev = gram_eigs(rows).clamp(min=0).sort(descending=True).values
    return float(ev[:16].sum())
bf = {}
for slot in SLOTS:
    key = f"L{slot[0]} {slot[1]}"
    A, B = load_factors([files_497[i] for i in single], slot)
    Uf, S, Vf = factors(A, B)
    W = base_w(f"model.layers.{slot[0]}.self_attn.{slot[1]}_proj.weight")
    Uw, _, Vwh = torch.linalg.svd(W, full_matrices=False)
    bf[key] = {}
    for side in ("in", "out"):
        F = Vf if side == "in" else Uf
        Bw = Vwh.T if side == "in" else Uw
        rows = side_rows(F, S)
        nulls = [side_rows(rand_frames(*F.shape, gen), S) for _ in range(args.repeats)]
        E = float(rows.pow(2).sum())
        En = [float(r.pow(2).sum()) for r in nulls]
        out = dict(d=int(F.shape[1]), top16_share_before=top16_energy(rows) / E,
                   null_top16_share_before=float(np.mean([top16_energy(r) / e for r, e in zip(nulls, En)])))
        out["excess_before"] = out["top16_share_before"] - out["null_top16_share_before"]
        for k in (16, 128):
            Q = Bw[:, :k]
            proj = lambda X: X - (X @ Q) @ Q.T
            Eb = float((rows @ Q).pow(2).sum())
            T = top16_energy(proj(rows))
            Tn, Ebn = [], []
            for r in nulls:
                Ebn.append(float((r @ Q).pow(2).sum()))
                Tn.append(top16_energy(proj(r)))
            fixed_null = np.mean([t / e for t, e in zip(Tn, En)])
            renorm_null = np.mean([t / (e - eb) for t, e, eb in zip(Tn, En, Ebn)])
            out[f"base{k}"] = dict(
                energy_in_base_share=Eb / E, null_energy_in_base_share=float(np.mean([eb / e for eb, e in zip(Ebn, En)])),
                chance=k / F.shape[1],
                top16_after_fixed=T / E, null_top16_after_fixed=float(fixed_null),
                excess_after_fixed=T / E - float(fixed_null),
                top16_after_renorm=T / (E - Eb), null_top16_after_renorm=float(renorm_null),
                excess_after_renorm=T / (E - Eb) - float(renorm_null),
                null_sd_fixed=float(np.std([t / e for t, e in zip(Tn, En)])))
        bf[key][side] = out
        log("basefixed", key, side, "before", round(out["excess_before"], 3),
            "after128 fixed", round(out["base128"]["excess_after_fixed"], 3),
            "renorm", round(out["base128"]["excess_after_renorm"], 3),
            "E_base128", round(out["base128"]["energy_in_base_share"], 3))
    res["basefixed"] = bf
    save()
log("done")
