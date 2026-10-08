"""Is the real (seed-independent) shared structure just the base model's dominant directions?

Own-seed adapters only. For each slot:
- full-update shared subspaces (top-16 PCs of stacked s_j u_j / s_j v_j, uncentred and centred)
- null: same singular values, random orthonormal frames per adapter
- overlap of the adapters' shared subspace with the base weight's top-16 singular subspace
  (mean cos^2 of principal angles; chance = 16/d)
- share of total adapter update energy inside the base weight's top-k singular subspaces
- layer 0 input side: same against the top directions of the actual layer-0 inputs
  (RMSNorm(embedding) * norm weight over the vocabulary)
"""
import json
import numpy as np, torch
from safetensors import safe_open
exec(open("F:/uag/robust.py").read().split("out = {")[0].split("def spectrum")[0])
torch.set_num_threads(6)
idx = json.load(open("F:/mistral/model.safetensors.index.json"))["weight_map"]
def base(name):
    with safe_open("F:/mistral/" + idx[name], "pt") as h:
        return h.get_tensor(name).double()

def topk_pc(rows, k=16, centre=False):
    M = rows - rows.mean(0) if centre else rows
    _, s, vh = torch.linalg.svd(M, full_matrices=False)
    ev = s ** 2
    return vh[:k].T, float(ev[:k].sum() / ev.sum())

def overlap(P, Q):  # mean cos^2 of principal angles between column spaces
    return float((P.T @ Q).pow(2).sum() / P.shape[1])

def energy_in(basis, F, S):  # share of sum_i ||dW_i||^2 inside span(basis), side given by F
    return float(((basis.T @ F) * S[:, None, :]).pow(2).sum() / S.pow(2).sum())

g = torch.Generator().manual_seed(0)
def rand_frames(n, d, r):
    return torch.linalg.qr(torch.randn(n, d, r, generator=g, dtype=torch.float64))[0]

layer0_in = None
out = []
for l in (0, 8, 16, 24, 31):
    for m in "qkv":
        Uf, S, Vf = factors(*load((l, m), indep))
        W = base(f"model.layers.{l}.self_attn.{m}_proj.weight")
        Uw, Sw, Vwh = torch.linalg.svd(W, full_matrices=False)
        res = {"slot": f"L{l} {m}", "n": len(indep)}
        for side, F, Bw in (("out", Uf, Uw), ("in", Vf, Vwh.T)):
            rows = (F * S[:, None, :]).transpose(1, 2).reshape(-1, F.shape[1])
            P, v = topk_pc(rows)
            _, vc = topk_pc(rows, centre=True)
            Fr = rand_frames(F.shape[0], F.shape[1], F.shape[2])
            _, vn = topk_pc((Fr * S[:, None, :]).transpose(1, 2).reshape(-1, F.shape[1]))
            d = F.shape[1]
            res[side] = dict(top16=round(v, 3), top16_centred=round(vc, 3), top16_null=round(vn, 3),
                             overlap_base16=round(overlap(P, Bw[:, :16]), 3), chance=round(16 / d, 4),
                             energy_base16=round(energy_in(Bw[:, :16], F, S), 3),
                             energy_base128=round(energy_in(Bw[:, :128], F, S), 3),
                             chance128=round(128 / d, 4))
            if l == 0 and side == "in":
                if layer0_in is None:
                    E = base("model.embed_tokens.weight")
                    x = E / E.pow(2).mean(1, keepdim=True).add(1e-5).sqrt() * base("model.layers.0.input_layernorm.weight")
                    layer0_in = torch.linalg.svd(x, full_matrices=False)[2].T
                res[side]["overlap_inputs16"] = round(overlap(P, layer0_in[:, :16]), 3)
                res[side]["energy_inputs16"] = round(energy_in(layer0_in[:, :16], F, S), 3)
                res[side]["energy_inputs128"] = round(energy_in(layer0_in[:, :128], F, S), 3)
        out.append(res)
        print(json.dumps(res), flush=True)
json.dump(out, open("F:/uag/basecheck.json", "w"))
