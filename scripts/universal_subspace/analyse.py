"""Universal-subspace analysis of the paper's exact Lots-of-LoRAs set (v3 Table 10).

usage: python analyse.py MANIFEST OUT.json [--base F:/mistral] [--quick]

Sections (all results saved to OUT.json):
  grouping   inferred shared-initialisation groups: cosine distributions, threshold sweep,
             agreement across all 15 slots, moments of the group mean, distance from group mean
  paper      the paper's centred A/B spectra on the full set; no-training simulations
  matched    shared group vs equally many singletons; size-matched random null (repeats);
             singleton subsample spread
  gauge      uncentred (gauge-invariant) one-sided spectra of dW = BA; strength-matched null
             (repeated rotations), adapter bootstrap, unit-Frobenius variant
  base       overlap of singleton output/input subspaces with base-weight singular subspaces;
             excess over the null before and after removing base-aligned components
  jd         Compress-then-Serve JD-Full reconstruction (mean relative squared Frobenius error),
             shared vs singletons vs mixed; init sensitivity, restarts, exact vs low-rank SVD,
             convergence trace
"""
import argparse
import json
import time

import numpy as np
import torch
from safetensors import safe_open

from common import (SLOTS, components, cos_matrix, factors, gram_eigs, kaiming_like, kurtosis,
                    load_factors, load_manifest, paper_spectrum, quantiles, rand_frames, share_curve,
                    side_rows, uncentred_spectrum)

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("out")
ap.add_argument("--base", default="F:/mistral")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--quick", action="store_true", help="few slots, few repeats (smoke test)")
args = ap.parse_args()
torch.set_num_threads(12)
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
rng = np.random.default_rng(0)
slots = SLOTS[:2] if args.quick else SLOTS
R = 2 if args.quick else args.repeats
ids, files = load_manifest(args.manifest)
N = len(files)
res = {"manifest": args.manifest, "n_adapters": N, "ids": ids, "threshold": args.threshold,
       "repeats": R, "torch_seed": 0, "numpy_seed": 0, "slots": [f"L{l} {m}" for l, m in slots]}
t0 = time.time()
def log(*a):
    print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
def save():
    json.dump(res, open(args.out, "w"))

# ---------------------------------------------------------------- grouping
A0, _ = load_factors(files, (0, "q"))
C = cos_matrix(A0)
lab = components(C, args.threshold)
sizes = np.bincount(lab)
big = int(sizes.argmax())
shared = np.where(lab == big)[0]
single = np.where(sizes[lab] == 1)[0]
iu = torch.triu_indices(N, N, 1)
pair = C[iu[0], iu[1]]
same = torch.as_tensor(lab[iu[0]] == lab[iu[1]])
sweep = {}
for th in (0.15, 0.2, 0.3, 0.4, 0.5, 0.6):
    l2 = components(C, th)
    s2 = np.bincount(l2)
    sweep[str(th)] = dict(groups=int(len(s2)), largest=int(s2.max()), singletons=int((s2 == 1).sum()),
                          same_partition=bool(len(s2) == len(sizes) and (np.sort(s2) == np.sort(sizes)).all()))
agree = {}
for slot in slots:
    A, _ = load_factors(files, slot)
    l2 = components(cos_matrix(A), args.threshold)
    agree[f"L{slot[0]} {slot[1]}"] = bool(((l2[:, None] == l2[None, :]) == (lab[:, None] == lab[None, :])).all())
Cs = C[np.ix_(shared, shared)]
res["grouping"] = dict(
    method=f"connected components of layer-0 q_proj A cosine > {args.threshold}",
    n_groups=int(len(sizes)), sizes_desc=sorted(sizes.tolist(), reverse=True)[:10],
    largest_group=int(len(shared)), singletons=int(len(single)), multi_member_groups=int((sizes > 1).sum()),
    within_pair_cos=quantiles(pair[same]), between_pair_cos=quantiles(pair[~same]),
    largest_group_min_pairwise_cos=float(Cs[~torch.eye(len(shared), dtype=bool)].min()),
    threshold_sweep=sweep, same_grouping_in_slot=agree,
    shared_ids=[ids[i] for i in shared])
log("grouping", res["grouping"]["n_groups"], "groups, largest", len(shared), "singletons", len(single))

# moments of the group mean and distance from it, per slot (A only)
mom = {}
for slot in slots:
    A, _ = load_factors([files[i] for i in shared], slot)
    m = A.mean(0)
    b = 1 / np.sqrt(A.shape[-1])
    d = (A - m).flatten(1).norm(dim=1) / m.flatten().norm()
    mom[f"L{slot[0]} {slot[1]}"] = dict(std=float(m.std()), std_uniform=float(b / np.sqrt(3)),
                                        kurtosis=kurtosis(m), max_abs=float(m.abs().max()), bound=float(b),
                                        dist_from_group_mean_rel=quantiles(d, (0.1, 0.5, 0.9)))
res["group_mean_moments"] = mom
save(); log("moments done")

# ---------------------------------------------------------------- paper-style spectra + simulations
n = min(len(shared), len(single))
sub_shared = rng.choice(shared, n, replace=False)
res["matched_n"] = int(n)
res["matched_rank_cap_centred"] = int(n * 16 - 1)
paper, matched, gauge, base = [], [], [], []
idx = json.load(open(f"{args.base}/model.safetensors.index.json"))["weight_map"]
def base_w(name):
    with safe_open(f"{args.base}/" + idx[name], "pt") as h:
        return h.get_tensor(name).double()

for slot in slots:
    key = f"L{slot[0]} {slot[1]}"
    A, B = load_factors(files, slot)
    Bt = B.transpose(1, 2)
    r = {"slot": key, "A_full": paper_spectrum(A.reshape(-1, A.shape[-1])),
         "B_full": paper_spectrum(Bt.reshape(-1, Bt.shape[-1]))}
    # no-training simulations with the same grouping: pure inits; inits + movement of the measured size
    sims_pure, sims_moved = [], []
    gm = A[shared].mean(0)
    dist = (A - gm).flatten(1).norm(dim=1) / gm.flatten().norm()
    med = float(dist[shared].median())
    for _ in range(R):
        inits = kaiming_like((len(sizes),) + A.shape[1:], gen)[lab]
        sims_pure.append(paper_spectrum(inits.reshape(-1, A.shape[-1]))["top16"])
        noise = torch.randn(A.shape, generator=gen, dtype=torch.float64)
        noise = noise / noise.flatten(1).norm(dim=1)[:, None, None] * inits.flatten(1).norm(dim=1)[:, None, None] * med
        sims_moved.append(paper_spectrum((inits + noise).reshape(-1, A.shape[-1]))["top16"])
    r["sim_no_training_top16"] = dict(mean=float(np.mean(sims_pure)), sd=float(np.std(sims_pure)))
    r["sim_init_plus_movement_top16"] = dict(mean=float(np.mean(sims_moved)), sd=float(np.std(sims_moved)),
                                             relative_movement=med)
    paper.append(r)

    # matched: shared group vs singletons; size-matched null; singleton subsample spread
    mr = {"slot": key}
    for nm, X in (("A", A), ("B", Bt)):
        mr[nm + "_shared"] = paper_spectrum(X[sub_shared].reshape(-1, X.shape[-1]))
        singles = []
        for _ in range(R):
            s = rng.choice(single, n, replace=False)
            singles.append(paper_spectrum(X[s].reshape(-1, X.shape[-1]))["top16"])
        mr[nm + "_singletons_top16"] = dict(mean=float(np.mean(singles)), sd=float(np.std(singles)),
                                            min=float(np.min(singles)), max=float(np.max(singles)))
        if nm == "A":
            nulls = [paper_spectrum(kaiming_like((n,) + A.shape[1:], gen).reshape(-1, A.shape[-1]))["top16"]
                     for _ in range(R)]
            mr["A_null_independent_inits_top16"] = dict(mean=float(np.mean(nulls)), sd=float(np.std(nulls)))
    s_fixed = rng.choice(single, n, replace=False)
    mr["A_singletons_example"] = paper_spectrum(A[s_fixed].reshape(-1, A.shape[-1]))
    matched.append(mr)

    # gauge-invariant uncentred one-sided spectra; strength-matched null, bootstrap, unit norm
    W = base_w(f"model.layers.{slot[0]}.self_attn.{slot[1]}_proj.weight")
    Uw, _, Vwh = torch.linalg.svd(W, full_matrices=False)
    g = {"slot": key}
    bres = {"slot": key}
    fac = {nm: factors(A[ix], B[ix]) for nm, ix in (("shared", sub_shared), ("single", s_fixed))}
    for side in ("in", "out"):
        for nm in ("shared", "single"):
            Uf, S, Vf = fac[nm]
            F = Vf if side == "in" else Uf
            g[f"{side}_{nm}"] = uncentred_spectrum(side_rows(F, S))["top16"]
            Sn = S / S.norm(dim=1, keepdim=True)
            g[f"{side}_{nm}_unitnorm"] = uncentred_spectrum(side_rows(F, Sn))["top16"]
            nul, nuln = [], []
            for _ in range(R):
                Fr = rand_frames(*F.shape, gen)
                nul.append(uncentred_spectrum(side_rows(Fr, S))["top16"])
                nuln.append(uncentred_spectrum(side_rows(Fr, Sn))["top16"])
            g[f"{side}_{nm}_null"] = dict(mean=float(np.mean(nul)), sd=float(np.std(nul)))
            g[f"{side}_{nm}_unitnorm_null"] = dict(mean=float(np.mean(nuln)), sd=float(np.std(nuln)))
        # bootstrap over singleton adapters (excess over a fresh null each time)
        Uf, S, Vf = fac["single"]
        F = Vf if side == "in" else Uf
        ex = []
        for _ in range(R):
            b = torch.as_tensor(rng.integers(0, len(S), len(S)))
            real = uncentred_spectrum(side_rows(F[b], S[b]))["top16"]
            nul = uncentred_spectrum(side_rows(rand_frames(*F[b].shape, gen), S[b]))["top16"]
            ex.append(real - nul)
        g[f"{side}_single_excess_bootstrap"] = dict(mean=float(np.mean(ex)), sd=float(np.std(ex)),
                                                   min=float(np.min(ex)), max=float(np.max(ex)))
        # base alignment (singletons): overlap with base top-16, and excess after removing base top-k
        Bw = Uw if side == "out" else Vwh.T
        rows = side_rows(F, S)
        _, _, vh = torch.linalg.svd(rows, full_matrices=False)
        P = vh[:16].T
        d = F.shape[1]
        bres[side] = dict(overlap_base16=float((P.T @ Bw[:, :16]).pow(2).sum() / 16), chance=16 / d,
                          energy_in_base128=float((rows @ Bw[:, :128]).pow(2).sum() / rows.pow(2).sum()),
                          chance128=128 / d)
        for kk in (16, 128):
            Q = Bw[:, :kk]
            proj = lambda X: X - (X @ Q) @ Q.T
            real = uncentred_spectrum(proj(rows))["top16"]
            nul = np.mean([uncentred_spectrum(proj(side_rows(rand_frames(*F.shape, gen), S)))["top16"]
                           for _ in range(R)])
            bres[side][f"excess_after_removing_base{kk}"] = float(real - nul)
        bres[side]["excess_before"] = g[f"{side}_single"] - g[f"{side}_single_null"]["mean"]
    gauge.append(g)
    base.append(bres)
    res["paper"], res["matched"], res["gauge"], res["base"] = paper, matched, gauge, base
    save()
    log(key, "A_full", round(r["A_full"]["top16"], 3), "sim", round(r["sim_no_training_top16"]["mean"], 3),
        "| matched A", round(mr["A_shared"]["top16"], 3), round(mr["A_singletons_top16"]["mean"], 3),
        "null", round(mr["A_null_independent_inits_top16"]["mean"], 4),
        "| gauge in", round(g["in_shared"], 3), round(g["in_single"], 3), round(g["in_single_null"]["mean"], 3),
        "out", round(g["out_single"], 3), round(g["out_single_null"]["mean"], 3))

# ---------------------------------------------------------------- JD-Full compression
def jd_full(Uf, S, Vf, Rk, iters=10, init="input", exact=False, seed=0, trace=False):
    g2 = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    Wn = S / S.norm(dim=1, keepdim=True)
    Ufw = Uf * Wn[:, None, :]
    def top(M):
        if exact:
            return torch.linalg.svd(M, full_matrices=False)[0][:, :Rk]
        return torch.svd_lowrank(M, q=Rk + 16, niter=4)[0][:, :Rk]
    if init == "input":
        V = top((Vf * Wn[:, None, :]).transpose(0, 1).reshape(Vf.shape[1], -1))
    else:
        V = torch.linalg.qr(torch.randn(Vf.shape[1], Rk, generator=g2, dtype=torch.float64))[0]
    errs = []
    for _ in range(iters):
        U = top((Ufw @ (Vf.transpose(1, 2) @ V)).transpose(0, 1).reshape(Ufw.shape[1], -1))
        V = top((Vf @ (Ufw.transpose(1, 2) @ U)).transpose(0, 1).reshape(Vf.shape[1], -1))
        if trace:
            core = (U.T @ Ufw) @ (Vf.transpose(1, 2) @ V)
            errs.append(float((1 - (core ** 2).sum((1, 2))).mean()))
    core = (U.T @ Ufw) @ (Vf.transpose(1, 2) @ V)
    e = 1 - (core ** 2).sum((1, 2))   # relative squared Frobenius error per adapter (unit-norm dW_i)
    return (e, errs) if trace else e

jd = []
m = min(100, n)
jd_slots = [(0, "q"), (0, "v"), (16, "q"), (16, "v"), (31, "q"), (31, "v")] if not args.quick else slots[:1]
for slot in jd_slots:
    A, B = load_factors(files, slot)
    sh = factors(A[sub_shared[:m]], B[sub_shared[:m]])
    si = factors(A[s_fixed[:m]], B[s_fixed[:m]])
    mx = tuple(torch.cat([sh[j][:m // 2], si[j][:m // 2]]) for j in range(3))
    for Rk in (16, 32, 64):
        row = {"slot": f"L{slot[0]} {slot[1]}", "R": Rk, "metric": "mean relative squared Frobenius error"}
        for init in ("input", "random"):
            for nm, fac in (("shared", sh), ("single", si)):
                es = [float(jd_full(*fac, Rk, init=init, seed=s).mean()) for s in range(3 if init == "random" else 1)]
                row[f"{nm}_{init}"] = dict(mean=float(np.mean(es)), runs=es)
            e = jd_full(*mx, Rk, init=init)
            row[f"mixed_{init}"] = dict(shared_half=float(e[:m // 2].mean()), single_half=float(e[m // 2:].mean()))
        if slot == (16, "q") and Rk == 16:
            row["exact_svd"] = {nm: float(jd_full(*fac, Rk, exact=True).mean()) for nm, fac in (("shared", sh), ("single", si))}
            row["trace_20_iters"] = {nm: jd_full(*fac, Rk, iters=20, trace=True)[1] for nm, fac in (("shared", sh), ("single", si))}
        jd.append(row)
        res["jd"] = jd
        save()
        log("jd", row["slot"], Rk, {k: (round(v["mean"], 3) if "mean" in v else v) for k, v in row.items() if isinstance(v, dict) and "mean" in v})
log("done")
