"""Spread of the singleton adapters' excess over the strength-matched null (replaces the bootstrap
in analyse.py, which resampled with replacement: duplicated adapters share directions in the real
data but get independent random directions in the null, inflating the excess).

usage: python excess_subsample.py MANIFEST GROUPING.json OUT.json

For each slot and side: 20 subsamples of 80% of the singletons, without replacement. Excess =
top-16 share (uncentred, gauge-invariant) of the real adapters minus that of a fresh
strength-matched null (same singular values, random orthonormal directions). Raw and unit-norm.
"""
import json
import sys

import numpy as np
import torch

from common import SLOTS, factors, load_factors, load_manifest, rand_frames, side_rows, uncentred_spectrum

manifest, grouping, out = sys.argv[1:4]
torch.set_num_threads(12)
gen = torch.Generator().manual_seed(2)
rng = np.random.default_rng(2)
ids, files = load_manifest(manifest)
shared_ids = set(json.load(open(grouping))["grouping"]["shared_ids"])
single = np.array([i for i, x in enumerate(ids) if x not in shared_ids])
m = int(0.8 * len(single))
res = {"n_singletons": int(len(single)), "subsample": m, "repeats": 20, "seed": 2, "rows": []}
for slot in SLOTS:
    A, B = load_factors([files[i] for i in single], slot)
    Uf, S, Vf = factors(A, B)
    row = {"slot": f"L{slot[0]} {slot[1]}"}
    for side, F in (("in", Vf), ("out", Uf)):
        for norm in ("raw", "unit"):
            Sx = S if norm == "raw" else S / S.norm(dim=1, keepdim=True)
            ex = []
            for _ in range(20):
                ix = torch.as_tensor(rng.choice(len(single), m, replace=False))
                real = uncentred_spectrum(side_rows(F[ix], Sx[ix]))["top16"]
                nul = uncentred_spectrum(side_rows(rand_frames(m, F.shape[1], F.shape[2], gen), Sx[ix]))["top16"]
                ex.append(real - nul)
            row[f"{side}_{norm}"] = dict(mean=float(np.mean(ex)), sd=float(np.std(ex)),
                                         min=float(np.min(ex)), max=float(np.max(ex)))
    res["rows"].append(row)
    print(row["slot"], {k: (round(v["mean"], 3), round(v["sd"], 3)) for k, v in row.items() if k != "slot"}, flush=True)
    json.dump(res, open(out, "w"), indent=1)
print("done")
