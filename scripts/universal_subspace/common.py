"""Shared helpers for the universal-subspace analysis of Lots-of-LoRAs adapters."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

PFX = "base_model.model.model.layers.{}.self_attn.{}_proj.lora_{}.weight"
LAYERS = (0, 8, 16, 24, 31)
SLOTS = [(l, m) for l in LAYERS for m in "qkv"]  # a "slot" is one (layer, projection) pair


def load_manifest(path, roles=("pool", "iid"), root="F:/lol"):
    """Adapter files for a manifest written from the paper's Table 10 (roles: pool / iid / ood)."""
    man = json.load(open(path))["adapters"]
    keep = [a for a in man if a["role"] in roles]
    files = [f"{root}/{a['id'].split('/')[1]}/adapter_model.safetensors" for a in keep]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} adapters missing, e.g. {missing[0]}")
    return [a["id"] for a in keep], files


def load_factors(files, slot, dtype=torch.float64):
    """A (n, r, d_in) and B (n, d_out, r) for one slot. LoRA scaling alpha/r is constant (2) and omitted."""
    A, B = [], []
    for f in files:
        with safe_open(f, "pt") as h:
            A.append(h.get_tensor(PFX.format(*slot, "A")).to(dtype))
            B.append(h.get_tensor(PFX.format(*slot, "B")).to(dtype))
    return torch.stack(A), torch.stack(B)


def cos_matrix(X):
    F = X.reshape(len(X), -1)
    F = F / F.norm(dim=1, keepdim=True)
    return F @ F.T


def components(C, threshold):
    """Connected components of the graph cos > threshold (single linkage)."""
    n = len(C)
    adj = (C > threshold).cpu().numpy()
    lab = -np.ones(n, int)
    g = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]
        while stack:
            j = stack.pop()
            if lab[j] < 0:
                lab[j] = g
                stack += list(np.where(adj[j] & (lab < 0))[0])
        g += 1
    return lab


def share_curve(ev, upto=128):
    ev = torch.as_tensor(ev).clamp(min=0).sort(descending=True).values
    c = torch.cumsum(ev, 0) / ev.sum()
    return dict(top16=float(c[15]), top64=float(c[63]), k90=int(torch.searchsorted(c, 0.9)) + 1,
                curve=[round(float(x), 5) for x in c[:upto]])


def gram_eigs(M):
    """Eigenvalues of M^T M via the smaller Gram matrix."""
    G = M @ M.T if M.shape[0] <= M.shape[1] else M.T @ M
    return torch.linalg.eigvalsh(G)


def paper_spectrum(rows):
    """The paper's analysis: stack rank vectors, centre feature-wise, PCA."""
    return share_curve(gram_eigs(rows - rows.mean(0)))


def uncentred_spectrum(rows):
    """Gauge-invariant: for rows s_j v_j this is the spectrum of sum_i dW_i^T dW_i."""
    return share_curve(gram_eigs(rows))


def factors(A, B):
    """Compact SVD of each B_i A_i via QR: Uf (n,dout,r), S (n,r), Vf (n,din,r)."""
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.transpose(1, 2))
    u, s, vh = torch.linalg.svd(Rb @ Ra.transpose(1, 2))
    return Qb @ u, s, Qa @ vh.transpose(1, 2)


def side_rows(F, S):
    """Stack s_j f_j for every adapter: (n*r, d)."""
    return (F * S[:, None, :]).transpose(1, 2).reshape(-1, F.shape[1])


def rand_frames(n, d, r, gen):
    return torch.linalg.qr(torch.randn(n, d, r, generator=gen, dtype=torch.float64))[0]


def kaiming_like(shape, gen):
    """PEFT's default LoRA A init: kaiming_uniform_(a=sqrt(5)) = U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    b = 1 / math.sqrt(shape[-1])
    return (torch.rand(shape, generator=gen, dtype=torch.float64) * 2 - 1) * b


def kurtosis(x):
    x = x.flatten().double()
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def quantiles(x, qs=(0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)):
    x = torch.as_tensor(x).double().flatten()
    return {str(q): round(float(torch.quantile(x, q)), 5) for q in qs}
