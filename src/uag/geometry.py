"""Stage 1: characterise the geometry of effective updates (spec §6).

Seed invariance, dataset invariance (is an update closer by transformation identity than
by dataset identity?), layer localisation, base-distance dependence, and the negative
controls (shuffled task pairing, rotated coordinates, norm-matched random updates, maps
fitted on random task subsets). All functions operate on :class:`DeltaSet` objects and
never on raw LoRA factors.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Callable, Hashable, Sequence

import numpy as np

from .extract_delta import DeltaSet, LowRank
from .spectral import cosine, left_right_overlap, singular_values, spectrum_similarity


def module_similarity(a: LowRank, b: LowRank) -> dict[str, float]:
    """Same-shape similarity suite: cosine, subspace overlaps, spectrum similarity."""
    out = {"cosine": cosine(a, b), "spectrum_similarity": spectrum_similarity(singular_values(a), singular_values(b))}
    out.update(left_right_overlap(a, b))
    return out


def deltaset_similarity(a: DeltaSet, b: DeltaSet) -> dict[str, float]:
    """Module-averaged similarity plus the global (all-module) cosine."""
    names = sorted(set(a.modules) & set(b.modules))
    per = [module_similarity(a.modules[n], b.modules[n]) for n in names]
    out = {k: float(np.mean([p[k] for p in per])) for k in per[0]} if per else {}
    num = sum(cosine(a.modules[n], b.modules[n]) * a.modules[n].fro() * b.modules[n].fro() for n in names)
    out["global_cosine"] = float(num / (a.fro() * b.fro())) if a.fro() * b.fro() > 0 else 0.0
    return out


def seed_invariance(runs: Sequence[DeltaSet]) -> dict[str, Any]:
    """Pairwise similarity across seeds of one base × task (5-10 seeds recommended)."""
    pairs = [deltaset_similarity(a, b) for a, b in itertools.combinations(runs, 2)]
    if not pairs:
        return {"n_pairs": 0}
    return {"n_pairs": len(pairs), **{k: {"mean": float(np.mean([p[k] for p in pairs])),
                                          "std": float(np.std([p[k] for p in pairs]))} for k in pairs[0]}}


def identity_test(deltas: dict[tuple[Hashable, Hashable], DeltaSet], metric: str = "global_cosine") -> dict[str, Any]:
    """Dataset invariance: keys are (transformation, dataset). Compares similarity of pairs
    sharing the transformation (different datasets) vs sharing the dataset (different
    transformations). Transformation identity should dominate if the geometry is semantic."""
    same_task, same_data, neither = [], [], []
    for (ka, a), (kb, b) in itertools.combinations(deltas.items(), 2):
        s = deltaset_similarity(a, b)[metric]
        if ka[0] == kb[0] and ka[1] != kb[1]:
            same_task.append(s)
        elif ka[1] == kb[1] and ka[0] != kb[0]:
            same_data.append(s)
        elif ka[0] != kb[0] and ka[1] != kb[1]:
            neither.append(s)
    mean = lambda x: float(np.mean(x)) if x else float("nan")  # noqa: E731
    return {"same_transformation": mean(same_task), "same_dataset": mean(same_data), "neither": mean(neither),
            "transformation_dominates": mean(same_task) > mean(same_data),
            "n": {"same_transformation": len(same_task), "same_dataset": len(same_data), "neither": len(neither)}}


def layer_localization(ds: DeltaSet, n_bins: int = 3) -> dict[str, Any]:
    """Share of update energy in early/middle/late layers and per module class."""
    layers = [ds.info[n]["layer"] for n in ds.modules]
    L = max(layers) + 1
    by_bin: dict[str, float] = defaultdict(float)
    by_cls: dict[str, float] = defaultdict(float)
    by_group: dict[str, float] = defaultdict(float)
    names = ["early", "middle", "late"] if n_bins == 3 else [f"bin{i}" for i in range(n_bins)]
    for n, lr in ds.modules.items():
        e = lr.fro() ** 2
        b = min(int(ds.info[n]["layer"] * n_bins / L), n_bins - 1)
        by_bin[names[b]] += e
        cls = ds.info[n]["cls"]
        by_cls[cls] += e
        by_group["attention" if cls in ("q", "k", "v", "o") else "mlp"] += e
    tot = sum(by_bin.values()) or 1.0
    norm = lambda d: {k: v / tot for k, v in d.items()}  # noqa: E731
    return {"by_depth": norm(by_bin), "by_class": norm(by_cls), "by_group": norm(by_group)}


def transfer_by_layer(reconstruction_modules: dict[str, dict[str, float]], info: dict[str, dict[str, Any]],
                      n_bins: int = 3) -> dict[str, float]:
    """Mean predicted-vs-true cosine per depth bin (where does transferable signal live?)."""
    L = max(v["layer"] for v in info.values()) + 1
    acc: dict[int, list[float]] = defaultdict(list)
    for n, rep in reconstruction_modules.items():
        acc[min(int(info[n]["layer"] * n_bins / L), n_bins - 1)].append(rep["cosine"])
    return {f"bin{b}": float(np.mean(v)) for b, v in sorted(acc.items())}


def base_distance_table(results: list[dict[str, Any]], features: Callable[[str, str], dict[str, float]]) -> dict:
    """Relate transfer (e.g. RecoveredLift per ordered pair) to base-pair features such as
    same_family, hidden-size ratio, tokenizer overlap or representation similarity.
    ``results`` rows need keys source, target, value. Returns Spearman correlations."""
    from scipy.stats import spearmanr

    rows = [{**features(r["source"], r["target"]), "value": r["value"]} for r in results]
    keys = [k for k in rows[0] if k != "value"] if rows else []
    out = {}
    for k in keys:
        x, y = [r[k] for r in rows], [r["value"] for r in rows]
        rho, p = spearmanr(x, y) if len(set(x)) > 1 else (float("nan"), float("nan"))
        out[k] = {"spearman_rho": float(rho), "p": float(p), "n": len(rows)}
    return out


# -- controls ------------------------------------------------------------------------


def shuffled_pairing(target: dict[str, DeltaSet], seed: int = 0) -> dict[str, DeltaSet]:
    """Control: pair each source task with a *different* target task (a derangement)."""
    keys = sorted(target)
    rng = np.random.default_rng(seed)
    while True:
        perm = list(rng.permutation(keys))
        if all(a != b for a, b in zip(keys, perm)) or len(keys) < 2:
            break
    return {k: target[p] for k, p in zip(keys, perm)}


def rotate_update(ds: DeltaSet, seed: int = 0) -> DeltaSet:
    """Control: rotate each module's left/right singular subspaces randomly (norm and
    spectrum preserved, geometry destroyed)."""
    rng = np.random.default_rng(seed)
    mods = {}
    for n, lr in ds.modules.items():
        U = np.linalg.qr(rng.normal(size=(lr.shape[0], max(lr.rank, 1))))[0][:, :lr.rank]
        V = np.linalg.qr(rng.normal(size=(lr.shape[1], max(lr.rank, 1))))[0][:, :lr.rank]
        mods[n] = LowRank(U, lr.S.copy(), V)
    return DeltaSet({**ds.meta, "control": "rotated"}, mods, ds.info)


def random_task_subsets(train_tasks: Sequence[str], size: int, n: int, seed: int = 0) -> list[tuple[str, ...]]:
    """Control: maps fitted on random subsets of the training tasks (deterministic)."""
    rng = np.random.default_rng(seed)
    return [tuple(sorted(rng.choice(list(train_tasks), size=size, replace=False))) for _ in range(n)]
