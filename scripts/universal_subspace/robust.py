"""Robustness checks for the universal-subspace critique.

(2) Full update dW = B A, gauge-invariant: per adapter, compact SVD dW = U S V^T; stack the
    16 input-side vectors s_j v_j (d_in) and output-side vectors s_j u_j (d_out) as in the paper,
    centre, PCA. Shared-seed vs own-seed adapters, matched n.
(3) Compress then Serve, JD-Full: shared orthonormal U, V (rank R) with dW_i ~ U U^T dW_i V V^T,
    each dW_i normalised to unit Frobenius norm, 10 alternating iterations.
    Relative error per adapter = 1 - ||U^T dW_i V||^2. Shared-seed vs own-seed, and a mixed set.
"""
import json, sys
import numpy as np, torch
from safetensors import safe_open

info = json.load(open("F:/uag/subspace_N500.json"))
lab = np.array(info["labels"]); sizes = np.bincount(lab)
files = info["files"]
big = sizes.argmax()
rng = np.random.default_rng(2)
shared_all = np.where(lab == big)[0]
indep_all = np.where(sizes[lab] == 1)[0]
n = min(len(shared_all), len(indep_all))
shared = rng.choice(shared_all, n, replace=False)
indep = rng.choice(indep_all, n, replace=False)
PFX = "base_model.model.model.layers.{}.self_attn.{}_proj.lora_{}.weight"
SLOTS = [(0, "q"), (0, "v"), (16, "q"), (16, "v"), (31, "q"), (31, "v")]
torch.set_num_threads(12)

def load(slot, idx):
    A, B = [], []
    for i in idx:
        with safe_open(files[i], "pt") as h:
            A.append(h.get_tensor(PFX.format(*slot, "A")).double())
            B.append(h.get_tensor(PFX.format(*slot, "B")).double())
    return torch.stack(A), torch.stack(B)  # n x r x din, n x dout x r

def factors(A, B):
    """Compact SVD of each B_i A_i via QR: returns Uf (n,dout,r), S (n,r), Vf (n,din,r)."""
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.transpose(1, 2))
    u, s, vh = torch.linalg.svd(Rb @ Ra.transpose(1, 2))
    return Qb @ u, s, Qa @ vh.transpose(1, 2)

def spectrum(rows):
    M = rows - rows.mean(0)
    ev = torch.linalg.svdvals(M) ** 2
    c = torch.cumsum(ev, 0) / ev.sum()
    return dict(top16=round(float(c[15]), 3), k90=int(torch.searchsorted(c, 0.9)) + 1)

def jd_full(Uf, S, Vf, R, iters=10):
    W = S / S.norm(dim=1, keepdim=True)            # unit Frobenius norm per adapter
    Ufw = Uf * W[:, None, :]                       # dW_i = Ufw_i Vf_i^T
    # init V from the stacked input sides
    top = lambda M: torch.svd_lowrank(M, q=R + 16, niter=4)[0][:, :R]
    V = top((Vf * W[:, None, :]).transpose(0, 1).reshape(Vf.shape[1], -1))
    for _ in range(iters):
        P = (Ufw @ (Vf.transpose(1, 2) @ V)).transpose(0, 1).reshape(Ufw.shape[1], -1)  # dW_i V
        U = top(P)
        Q = (Vf @ (Ufw.transpose(1, 2) @ U)).transpose(0, 1).reshape(Vf.shape[1], -1)   # dW_i^T U
        V = top(Q)
    core = (U.T @ Ufw) @ (Vf.transpose(1, 2) @ V)
    return 1 - (core ** 2).sum((1, 2))             # relative squared error per adapter

out = {"n": int(n), "full_update": [], "jd": []}
for slot in SLOTS:
    res = {"slot": f"L{slot[0]} {slot[1]}"}
    fac = {}
    for name, idx in (("shared", shared), ("indep", indep)):
        Uf, S, Vf = factors(*load(slot, idx))
        fac[name] = (Uf, S, Vf)
        res[name + "_in"] = spectrum((Vf * S[:, None, :]).transpose(1, 2).reshape(-1, Vf.shape[1]))
        res[name + "_out"] = spectrum((Uf * S[:, None, :]).transpose(1, 2).reshape(-1, Uf.shape[1]))
    out["full_update"].append(res)
    print(json.dumps(res), flush=True)
    # JD-Full compression, 100 adapters per set, several ranks
    m = 100
    mixed = tuple(torch.cat([fac["shared"][j][:m // 2], fac["indep"][j][:m // 2]]) for j in range(3))
    for R in (16, 32, 64):
        es = jd_full(*(t[:m] for t in fac["shared"]), R)
        ei = jd_full(*(t[:m] for t in fac["indep"]), R)
        em = jd_full(*mixed, R)
        r = {"slot": res["slot"], "R": R, "shared_err": round(float(es.mean()), 3),
             "indep_err": round(float(ei.mean()), 3),
             "mixed_err_shared_half": round(float(em[:m // 2].mean()), 3),
             "mixed_err_indep_half": round(float(em[m // 2:].mean()), 3)}
        out["jd"].append(r)
        print(json.dumps(r), flush=True)
json.dump(out, open("F:/uag/robust.json", "w"))
