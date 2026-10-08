"""Is the 'universal subspace' of Lots-of-LoRAs A matrices a shared-initialisation artefact?

1. Seed groups: cluster adapters by cosine of their A matrices.
2. Movement: how far each A sits from its group's mean (proxy for the shared init).
3. The paper's analysis (stack N*r rank rows per layer, centre, PCA) on
   raw A, A minus its group mean, B, and random-init null models.
"""
import glob, json, sys
import numpy as np
from safetensors import safe_open

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
files = sorted(glob.glob("F:/lol/*/adapter_model.safetensors"))[:N]
N = len(files)
LAYERS = [0, 8, 16, 24, 31]
PFX = "base_model.model.model.layers.{}.self_attn.{}_proj.lora_{}.weight"
keys = [(l, m, ab) for l in LAYERS for m in "qkv" for ab in "AB"]

data = {}
for k in keys:
    data[k] = None
for i, f in enumerate(files):
    with safe_open(f, "np") as h:
        for k in keys:
            x = h.get_tensor(PFX.format(*k)).astype(np.float32)
            if k[2] == "B":
                x = x.T  # rank rows: r x d_out
            if data[k] is None:
                data[k] = np.empty((N,) + x.shape, np.float32)
            data[k][i] = x
print(f"loaded {N} adapters", flush=True)

def cos_matrix(X):
    F = X.reshape(len(X), -1)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    return F @ F.T

# 1. seed groups from layer-0 q A
C = cos_matrix(data[(0, "q", "A")])
lab = -np.ones(N, int)
g = 0
for i in range(N):
    if lab[i] < 0:
        stack = [i]
        while stack:
            j = stack.pop()
            if lab[j] < 0:
                lab[j] = g
                stack += list(np.where((C[j] > 0.5) & (lab < 0))[0])
        g += 1
sizes = np.bincount(lab)
same = lab[:, None] == lab[None, :]
off = ~np.eye(N, dtype=bool)
print(f"\n== seed groups (layer 0 q A, cos > 0.5): {g} groups; sizes {sorted(sizes, reverse=True)[:20]}")
print(f"   singletons {np.sum(sizes == 1)}; largest group {sizes.max()} ({sizes.max() / N:.0%})")
print(f"   within-group cos: min {C[same & off].min():.3f} median {np.median(C[same & off]):.3f}")
print(f"   between-group cos: max {np.abs(C[~same]).max():.3f} median {np.median(np.abs(C[~same])):.4f}")
# consistency of grouping across layers/modules
for k in [(31, "v", "A"), (16, "k", "A")]:
    Ck = cos_matrix(data[k])
    agree = np.mean((Ck[off] > 0.5) == same[off])
    print(f"   grouping agreement with {k}: {agree:.4f}")

multi = np.isin(lab, np.where(sizes >= 2)[0])

def group_resid(X):
    R = X.copy()
    for gg in np.where(sizes >= 2)[0]:
        idx = lab == gg
        R[idx] -= X[idx].mean(0)
    return R

# 2. movement
X = data[(0, "q", "A")]
R = group_resid(X)
rel = np.linalg.norm(R[multi].reshape(multi.sum(), -1), axis=1) / np.linalg.norm(
    (X - R)[multi].reshape(multi.sum(), -1), axis=1)
print(f"\n== A distance from group mean, relative (layer 0 q): median {np.median(rel):.3f}, "
      f"10-90% {np.percentile(rel, 10):.3f}-{np.percentile(rel, 90):.3f}")

# 3. paper-style spectrum
def spectrum(X):
    M = X.reshape(-1, X.shape[-1]).astype(np.float64)
    M = M - M.mean(0)
    ev = np.linalg.eigvalsh(M.T @ M)[::-1].clip(0)
    c = np.cumsum(ev) / ev.sum()
    return dict(top16=c[15], top64=c[63], k90=int(np.searchsorted(c, 0.9)) + 1)

rng = np.random.default_rng(0)
def kaiming(shape):
    b = 1 / np.sqrt(shape[-1])
    return rng.uniform(-b, b, shape).astype(np.float32)

rows = []
for k in keys:
    Xk = data[k]
    r = {"key": f"L{k[0]} {k[1]} {k[2]}", "raw_all": spectrum(Xk), "raw_multi": spectrum(Xk[multi])}
    if k[2] == "A":
        r["minus_group_mean"] = spectrum(group_resid(Xk)[multi])
        if k[1] == "q" and k[0] in (0, 16):
            inits = kaiming((g,) + Xk.shape[1:])
            r["null_shared_seeds_same_groups"] = spectrum(inits[lab] + 0.05 * kaiming(Xk.shape))
            r["null_independent_inits"] = spectrum(kaiming(Xk.shape))
    rows.append(r)
    print(json.dumps(r), flush=True)

json.dump({"N": N, "groups": g, "sizes": sizes.tolist(), "labels": lab.tolist(),
           "files": files, "rows": rows}, open(f"F:/uag/subspace_N{N}.json", "w"))

# 4. matched control: shared-seed group vs the same number of independent-seed adapters
bigg = sizes.argmax()
shared = np.where(lab == bigg)[0]
indep = np.where(sizes[lab] == 1)[0]
n = min(len(shared), len(indep))
rs = np.random.default_rng(1)
shared, indep = rs.choice(shared, n, replace=False), rs.choice(indep, n, replace=False)
print(f"\n== matched control: {n} shared-seed vs {n} independent-seed adapters", flush=True)
matched = []
for k in keys:
    a, b = spectrum(data[k][shared]), spectrum(data[k][indep])
    matched.append({"key": f"L{k[0]} {k[1]} {k[2]}", "shared": a, "independent": b})
    print(f"L{k[0]:<2} {k[1]} {k[2]}  shared: top16 {a['top16']:.2f} k90 {a['k90']:4d}   "
          f"independent: top16 {b['top16']:.2f} k90 {b['k90']:4d}   (max rank {n * 16})", flush=True)
json.dump({"N": N, "groups": g, "sizes": sizes.tolist(), "labels": lab.tolist(), "files": files,
           "rows": rows, "matched": matched, "n_matched": int(n)}, open(f"F:/uag/subspace_N{N}.json", "w"))
