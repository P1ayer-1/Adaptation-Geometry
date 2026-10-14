"""Supplementary checks for the universal-subspace note.

usage: python supplement.py MANIFEST GROUPING.json OUT.json

  residual  shared group: A_i minus the group mean (an estimate of the shared init, which also
            absorbs any common learned change). Paper-style spectrum vs a strength-matched null
            (each residual keeps its singular values, gets random orthonormal directions).
  strength  what the strength-matched null is made of: per-adapter effective rank of dW = BA
            (participation ratio (sum s^2)^2 / sum s^4), share of the top singular value, and the
            spread of adapter norms ||dW||_F, for the shared group and the singletons.
  curves    cumulative variance curves (first 512 components) for Figure 1, slot L16 q, A:
            full set, no-training simulation, shared group, matched singletons, independent inits.
"""
import json
import sys

import numpy as np
import torch

from common import (SLOTS, factors, kaiming_like, load_factors, load_manifest, paper_spectrum,
                    quantiles, rand_frames, share_curve, gram_eigs)

manifest, grouping, out = sys.argv[1:4]
torch.set_num_threads(12)
gen = torch.Generator().manual_seed(1)
rng = np.random.default_rng(1)
ids, files = load_manifest(manifest)
G = json.load(open(grouping))
shared_ids = set(G["grouping"]["shared_ids"])
shared = np.array([i for i, x in enumerate(ids) if x in shared_ids])
others = np.array([i for i, x in enumerate(ids) if x not in shared_ids])
n = min(len(shared), len(others))
res = {"n_shared": int(len(shared)), "n_others": int(len(others)), "seed": 1, "residual": [], "strength": []}

for slot in SLOTS:
    key = f"L{slot[0]} {slot[1]}"
    A, B = load_factors(files, slot)
    # residual movement in the shared group
    Rm = A[shared] - A[shared].mean(0)
    real = paper_spectrum(Rm.reshape(-1, A.shape[-1]))
    u, s, vh = torch.linalg.svd(Rm, full_matrices=False)          # per adapter: 16 x 4096
    nul = []
    for _ in range(10):
        Vr = rand_frames(len(Rm), A.shape[-1], 16, gen).transpose(1, 2)
        nul.append(paper_spectrum(((u * s[:, None, :]) @ Vr).reshape(-1, A.shape[-1])))
    res["residual"].append(dict(slot=key, top16=real["top16"], k90=real["k90"],
                                null_top16=float(np.mean([x["top16"] for x in nul])),
                                null_top16_sd=float(np.std([x["top16"] for x in nul])),
                                null_k90=float(np.mean([x["k90"] for x in nul]))))
    # strength profile of dW
    st = {"slot": key}
    for nm, ix in (("shared", shared), ("others", others)):
        _, S, _ = factors(A[ix], B[ix])
        pr = S.pow(2).sum(1).pow(2) / S.pow(4).sum(1)
        top1 = S[:, 0].pow(2) / S.pow(2).sum(1)
        norm = S.pow(2).sum(1).sqrt()
        st[nm] = dict(effective_rank=quantiles(pr, (0.1, 0.5, 0.9)), top1_share=quantiles(top1, (0.1, 0.5, 0.9)),
                      norm=quantiles(norm, (0.1, 0.5, 0.9)), norm_max_over_median=float(norm.max() / norm.median()))
    res["strength"].append(st)
    print(key, res["residual"][-1], {k: (v["effective_rank"]["0.5"], v["norm_max_over_median"]) for k, v in st.items() if k != "slot"}, flush=True)
    if slot == (16, "q"):
        lab = np.full(len(ids), -1)
        lab[shared] = 0
        lab[others] = np.arange(1, len(others) + 1)
        inits = kaiming_like((len(others) + 1,) + A.shape[1:], gen)[lab]
        s_o = rng.choice(others, n, replace=False)
        s_s = rng.choice(shared, n, replace=False)
        curve = lambda X: share_curve(gram_eigs(X - X.mean(0)), upto=512)["curve"]
        res["curves_L16q_A"] = dict(
            full_set=curve(A.reshape(-1, A.shape[-1])),
            no_training_sim=curve(inits.reshape(-1, A.shape[-1])),
            shared_group=curve(A[s_s].reshape(-1, A.shape[-1])),
            singletons=curve(A[s_o].reshape(-1, A.shape[-1])),
            independent_inits=curve(kaiming_like((n,) + A.shape[1:], gen).reshape(-1, A.shape[-1])),
            n_matched=int(n))
    json.dump(res, open(out, "w"))
print("done")
