"""Spectral geometry of effective updates (spec §20.5, §14).

Frobenius norm, singular spectrum, effective rank, top-k singular vectors, principal angles,
and compact per-module summaries that are cached instead of dense ΔW.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .extract_delta import DeltaSet, LowRank


def frobenius(x: LowRank | np.ndarray) -> float:
    return x.fro() if isinstance(x, LowRank) else float(np.linalg.norm(x))


def singular_values(x: LowRank | np.ndarray) -> np.ndarray:
    return np.sort(x.S)[::-1] if isinstance(x, LowRank) else np.linalg.svd(x, compute_uv=False)


def effective_rank(s: np.ndarray, eps: float = 1e-12) -> float:
    """Roy & Vetterli (2007) entropy effective rank: exp(H(p)), p = s / sum(s)."""
    s = np.asarray(s, dtype=np.float64)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def stable_rank(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    return float((s ** 2).sum() / (s.max() ** 2)) if s.size and s.max() > 0 else 0.0


def energy_rank(s: np.ndarray, frac: float = 0.9) -> int:
    """Smallest k whose top-k singular values carry ``frac`` of the squared energy."""
    e = np.cumsum(np.sort(np.asarray(s, np.float64))[::-1] ** 2)
    return int(np.searchsorted(e / e[-1], frac) + 1) if e.size and e[-1] > 0 else 0


def top_k(x: LowRank | np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k (U, S, V) with deterministic signs."""
    lr = x if isinstance(x, LowRank) else LowRank.from_dense(x)
    order = np.argsort(-lr.S)[:k]
    U, S, V = lr.U[:, order], lr.S[order], lr.V[:, order]
    idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[idx, np.arange(U.shape[1])])
    signs[signs == 0] = 1
    return U * signs, S, V * signs


def orthonormal_basis(x: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(x, np.float64))
    return q


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between column spans of a and b (same ambient dim)."""
    qa, qb = orthonormal_basis(a), orthonormal_basis(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared cosine of principal angles in [0, 1] (1 = identical subspaces)."""
    ang = principal_angles(a, b)
    return float(np.mean(np.cos(ang) ** 2)) if ang.size else 0.0


def left_right_overlap(x: LowRank, y: LowRank, k: int | None = None) -> dict[str, float]:
    """Output-space and input-space subspace agreement of two same-shape updates."""
    k = k or min(x.rank, y.rank)
    ux, _, vx = top_k(x, k)
    uy, _, vy = top_k(y, k)
    return {"left_overlap": subspace_overlap(ux, uy), "right_overlap": subspace_overlap(vx, vy)}


def spectrum_similarity(s1: np.ndarray, s2: np.ndarray) -> float:
    """Cosine similarity of normalised, zero-padded singular spectra."""
    n = max(len(s1), len(s2))
    a = np.zeros(n)
    b = np.zeros(n)
    a[: len(s1)] = np.sort(s1)[::-1]
    b[: len(s2)] = np.sort(s2)[::-1]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def cosine(x: LowRank | np.ndarray, y: LowRank | np.ndarray) -> float:
    """Matrix cosine <X, Y>_F / (|X| |Y|), computed in factored form when possible."""
    if isinstance(x, LowRank) and isinstance(y, LowRank):
        inner = float(np.sum(((x.U.T @ y.U) * x.S[:, None]) * ((x.V.T @ y.V) * y.S[None, :])))
    else:
        xd = x.dense() if isinstance(x, LowRank) else x
        yd = y.dense() if isinstance(y, LowRank) else y
        inner = float(np.sum(xd * yd))
    denom = frobenius(x) * frobenius(y)
    return inner / denom if denom > 0 else 0.0


def relative_error(pred: LowRank | np.ndarray, ref: LowRank | np.ndarray) -> float:
    """||pred - ref||_F / ||ref||_F (Δ reconstruction error)."""
    p2, r2 = frobenius(pred) ** 2, frobenius(ref) ** 2
    cross = cosine(pred, ref) * frobenius(pred) * frobenius(ref)
    return float(np.sqrt(max(p2 + r2 - 2 * cross, 0.0)) / np.sqrt(r2)) if r2 > 0 else float("nan")


def module_summary(lr: LowRank, k: int = 16) -> dict[str, Any]:
    s = singular_values(lr)
    return {"shape": list(lr.shape), "fro": frobenius(lr), "spectral_norm": float(s[0]) if s.size else 0.0,
            "singular_values": s[:k].tolist(), "effective_rank": effective_rank(s),
            "stable_rank": stable_rank(s), "energy_rank_90": energy_rank(s, 0.9)}


def summarize_deltaset(ds: DeltaSet, k: int = 16) -> dict[str, Any]:
    """Compact spectral summary: per-module stats + energy by layer and module class."""
    modules = {n: {**ds.info.get(n, {}), **module_summary(lr, k)} for n, lr in ds.modules.items()}
    by_layer: dict[int, float] = defaultdict(float)
    by_cls: dict[str, float] = defaultdict(float)
    for m in modules.values():
        by_layer[m.get("layer", -1)] += m["fro"] ** 2
        by_cls[m.get("cls") or "?"] += m["fro"] ** 2
    total = sum(by_layer.values()) or 1.0
    return {"meta": ds.meta, "total_fro": float(np.sqrt(total)),
            "energy_by_layer": {str(k_): v / total for k_, v in sorted(by_layer.items())},
            "energy_by_class": {k_: v / total for k_, v in sorted(by_cls.items())},
            "modules": modules}


def save_spectral(ds: DeltaSet, out_dir: str | Path, k: int = 16) -> dict[str, Any]:
    """Cache the exact factored update (rank ≤ r, so compact) plus the summary JSON."""
    out = Path(out_dir)
    ds.save(out)
    summary = summarize_deltaset(ds, k)
    (out / "spectral_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
