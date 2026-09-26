"""Stage 3: compositionality of task latents vs weight-space baselines (spec §8).

Latent operations (sum, weighted sum, linear / spherical interpolation) produce a latent that
a :class:`~uag.latent_model.LatentModel` decodes into an update. Weight-space baselines act
directly on ΔW. Note that *sequential* application of two merged LoRA updates to the same
weights is exactly ``ΔW_A + ΔW_B`` (updates are additive), so it coincides with the linear
baseline; sequential application only differs when adapters are applied to activations of
different modules, which is not the case for merged LoRA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .extract_delta import DeltaSet

# -- latent operations ---------------------------------------------------------


def latent_sum(z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    return z_a + z_b


def latent_weighted(zs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    return sum(w * z for w, z in zip(weights, zs))


def lerp(z_a: np.ndarray, z_b: np.ndarray, alpha: float) -> np.ndarray:
    return (1 - alpha) * z_a + alpha * z_b


def slerp(z_a: np.ndarray, z_b: np.ndarray, alpha: float, eps: float = 1e-8) -> np.ndarray:
    """Spherical interpolation of directions with linearly interpolated norms."""
    a, b = z_a.ravel(), z_b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return lerp(z_a, z_b, alpha)
    ua, ub = a / na, b / nb
    omega = np.arccos(np.clip(ua @ ub, -1.0, 1.0))
    if omega < eps:
        direction = ua
    else:
        direction = (np.sin((1 - alpha) * omega) * ua + np.sin(alpha * omega) * ub) / np.sin(omega)
    return (((1 - alpha) * na + alpha * nb) * direction).reshape(z_a.shape)


# -- weight-space baselines ------------------------------------------------------


def delta_combine(dsets: Sequence[DeltaSet], weights: Sequence[float], task_id: str = "composed") -> DeltaSet:
    """Σ_i w_i ΔW_i per module (linear Δ_A + Δ_B baseline, LoRA-arithmetic, sequential merge)."""
    names = set(dsets[0].modules)
    for d in dsets[1:]:
        names &= set(d.modules)
    mods = {}
    for n in sorted(names):
        acc = None
        for d, w in zip(dsets, weights):
            term = d.modules[n].scaled(w)
            acc = term if acc is None else acc + term
        mods[n] = acc
    meta = {"kind": "composed", "task_id": task_id, "weights": list(weights),
            "components": [d.meta.get("task_id") for d in dsets]}
    return DeltaSet(meta, mods, {n: dsets[0].info.get(n, {}) for n in mods})


def norm_matched(ds: DeltaSet, reference: DeltaSet) -> DeltaSet:
    """Rescale ``ds`` to the reference's total Frobenius norm (the norm-matching control)."""
    c = reference.fro() / ds.fro() if ds.fro() > 0 else 0.0
    return DeltaSet({**ds.meta, "norm_matched_to": reference.meta.get("task_id")},
                    {n: lr.scaled(c) for n, lr in ds.modules.items()}, ds.info)


def fit_soup_weights(evaluate: Callable[[tuple[float, ...]], dict[str, float]],
                     grid: Sequence[tuple[float, ...]], objective: str = "min") -> dict:
    """LoRA-Soups-style mixing weights chosen on *validation* data by grid search.

    ``evaluate(weights) -> {factor: score}``; the objective is the minimum factor score
    (a composition must preserve every factor, not trade one away).
    """
    results = []
    for w in grid:
        scores = evaluate(tuple(w))
        agg = min(scores.values()) if objective == "min" else float(np.mean(list(scores.values())))
        results.append({"weights": tuple(w), "scores": scores, "objective": agg})
    best = max(results, key=lambda r: r["objective"])
    return {"best": best, "all": results}


def weight_grid(n: int = 2, values: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.25)) -> list[tuple[float, ...]]:
    import itertools

    return list(itertools.product(values, repeat=n))


def extrapolation_alphas() -> list[float]:
    """Negative-control interpolation coefficients, including values outside [0, 1]."""
    return [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


# -- evaluation ----------------------------------------------------------------------


@dataclass
class CompositionOutcome:
    factor_scores: dict[str, float]
    single_factor_refs: dict[str, float]
    thresholds: dict[str, float]

    @property
    def success(self) -> bool:
        """Succeeds only if *every* factor clears its threshold."""
        return all(self.factor_scores[f] >= self.thresholds[f] for f in self.thresholds)

    @property
    def retention(self) -> dict[str, float]:
        """Score relative to the corresponding single-factor adapter (1.0 = fully preserved)."""
        return {f: (self.factor_scores[f] / r if r else float("nan")) for f, r in self.single_factor_refs.items()}

    @property
    def traded_off(self) -> bool:
        """One factor preserved while another collapsed (a failure mode to report separately)."""
        ret = self.retention
        return max(ret.values()) >= 0.8 and min(ret.values()) < 0.5
