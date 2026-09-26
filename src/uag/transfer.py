"""Source→target pair maps and transfer baselines (spec §5.4, §20.7).

All learned maps are fitted per module on the *training* transformations of a
:class:`TaskSplit` only. Held-out task IDs are rejected at the fitting boundary
(:class:`HoldoutLeakError`), normalisation statistics come from training tasks only, and
every fitted map records its training and held-out task IDs.

Methods (all low capacity, so they cannot memorise task identities):

``procrustes_full``  same-shape two-sided orthogonal map  ΔW_t ≈ c·R_out ΔW_s R_inᵀ.
                     R acts as the identity outside the span of the training updates.
``svd_procrustes``   per-model truncated-SVD coordinate systems (from training updates);
                     orthogonal P, Q between k×k coordinates. Tolerates dimension mismatch.
``svd_linear``       same coordinates, general bilinear map C_t ≈ P C_s Qᵀ (ridge ALS).
``identity``         naïve copy (only where module shapes match; otherwise not applicable).
``cross_lora``       data-free Cross-LoRA-*style* projection through base-weight top-k
                     singular subspaces (an approximation of the published method).
``random``           norm-matched random low-rank update (negative control).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from safetensors.numpy import load_file, save_file

from .alignment import (ModuleCorrespondence, ModuleInfo, ModuleInventory, canonical_sign, match_modules,
                        procrustes)
from .config import MapSettings
from .extract_delta import DeltaSet, LowRank, from_factors

LEARNED_METHODS = ("procrustes_full", "svd_procrustes", "svd_linear")
BASELINE_METHODS = ("identity", "cross_lora", "random")
ALL_METHODS = LEARNED_METHODS + BASELINE_METHODS


class HoldoutLeakError(RuntimeError):
    """A held-out (or unknown) task tried to enter map fitting."""


@dataclass(frozen=True)
class TaskSplit:
    split_id: str
    train: tuple[str, ...]
    holdout: tuple[str, ...]

    def __post_init__(self):
        if set(self.train) & set(self.holdout):
            raise HoldoutLeakError(f"{self.split_id}: train and holdout overlap")
        if not self.train or not self.holdout:
            raise ValueError(f"{self.split_id}: need both training and held-out tasks")

    @staticmethod
    def from_holdout(split_id: str, all_tasks: list[str], holdout: list[str]) -> "TaskSplit":
        return TaskSplit(split_id, tuple(t for t in all_tasks if t not in holdout), tuple(holdout))

    def check_fit_inputs(self, task_ids, side: str) -> None:
        ids = set(task_ids)
        leaked = ids & set(self.holdout)
        if leaked:
            raise HoldoutLeakError(f"{self.split_id}: held-out tasks {sorted(leaked)} passed to {side} fitting")
        unknown = ids - set(self.train)
        if unknown:
            raise HoldoutLeakError(f"{self.split_id}: tasks {sorted(unknown)} are not declared training tasks")


def inventory_from_deltaset(ds: DeltaSet) -> ModuleInventory:
    mods = [ModuleInfo(n, int(ds.info[n]["layer"]), ds.info[n]["cls"], *lr.shape) for n, lr in ds.modules.items()]
    num_layers = int(ds.meta.get("num_layers") or (max(m.layer for m in mods) + 1))
    return ModuleInventory(ds.meta.get("model_type", "?"), num_layers, mods, ds.meta.get("module_omissions", {}))


# ---------------------------------------------------------------------------
# Low-rank orthogonal maps: R = I + W (R_small - I) Wᵀ
# ---------------------------------------------------------------------------


@dataclass
class SpanRotation:
    """Orthogonal map acting as the identity outside span(W)."""

    W: np.ndarray  # (d, m) orthonormal
    R: np.ndarray  # (m, m) orthogonal

    def apply(self, X: np.ndarray) -> np.ndarray:
        return X + self.W @ ((self.R - np.eye(self.R.shape[0])) @ (self.W.T @ X))

    @staticmethod
    def identity(d: int) -> "SpanRotation":
        return SpanRotation(np.zeros((d, 0)), np.zeros((0, 0)))


def span_procrustes(L: np.ndarray, Rf: np.ndarray, tol: float = 1e-10) -> SpanRotation:
    """Orthogonal maximiser of tr(Rᵀ M) for M = L Rfᵀ, restricted to span([L, Rf])."""
    # SVD (not unpivoted QR) so dependent stacked columns give the correct span.
    u, s, _ = np.linalg.svd(np.concatenate([L, Rf], axis=1), full_matrices=False)
    W = u[:, s > tol * max(s.max(initial=0.0), 1e-30)]
    if W.shape[1] == 0:
        return SpanRotation.identity(L.shape[0])
    return SpanRotation(W, procrustes((W.T @ L) @ (W.T @ Rf).T))


# ---------------------------------------------------------------------------
# Per-module fitting
# ---------------------------------------------------------------------------


@dataclass
class ModuleMap:
    source: str
    target: str
    method: str
    params: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)
    applicable: bool = True
    note: str = ""


def _norm(xs: list[LowRank]) -> float:
    n = float(np.mean([x.fro() for x in xs])) if xs else 0.0
    return n if n > 0 else 1.0


def _coordinate_basis(xs: list[LowRank], k: int, side: str) -> np.ndarray:
    """Top-k left (side='U') or right ('V') singular directions of the stacked training updates."""
    blocks = [(x.U if side == "U" else x.V) * x.S for x in xs if x.rank > 0]
    if not blocks:
        raise ValueError("no non-zero training updates for this module")
    m = np.concatenate(blocks, axis=1)
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    k_eff = int(min(k, np.sum(s > 1e-10 * s.max())))
    return canonical_sign(u[:, :k_eff])


def _fit_svd_coords(src: list[LowRank], tgt: list[LowRank], ns: float, nt: float, k: int):
    Us, Vs = _coordinate_basis(src, k, "U"), _coordinate_basis(src, k, "V")
    Ut, Vt = _coordinate_basis(tgt, k, "U"), _coordinate_basis(tgt, k, "V")
    Cs = [x.project(Us, Vs) / ns for x in src]
    Ct = [x.project(Ut, Vt) / nt for x in tgt]
    return Us, Vs, Ut, Vt, Cs, Ct


def _procrustes_objective(Cs, Ct, P, Q) -> float:
    return sum(float(np.sum(ct * (P @ cs @ Q.T))) for cs, ct in zip(Cs, Ct))


def _alt_procrustes_from(Cs, Ct, P, Q, iters: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(iters):
        P = procrustes(sum(ct @ Q @ cs.T for cs, ct in zip(Cs, Ct)))
        Q = procrustes(sum(ct.T @ P @ cs for cs, ct in zip(Cs, Ct)))
    return P, Q


def _alt_procrustes(Cs, Ct, iters: int, ridge: float = 1e-2, n_random: int = 4,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Orthogonal (semi-orthogonal if k differs) P, Q and scale c with Ct ≈ c P Cs Qᵀ.

    Alternating Procrustes is non-convex, so it is started from several deterministic
    initialisations (identity, polar projection of the bilinear least-squares fit, seeded
    random rotations) and the best objective is kept.
    """
    kt_l, ks_l = Ct[0].shape[0], Cs[0].shape[0]
    kt_r, ks_r = Ct[0].shape[1], Cs[0].shape[1]
    inits = [(np.eye(kt_l, ks_l), np.eye(kt_r, ks_r))]
    Pb, Qb = _alt_bilinear(Cs, Ct, np.eye(kt_l, ks_l), np.eye(kt_r, ks_r), ridge, iters)
    inits.append((procrustes(Pb), procrustes(Qb)))
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        inits.append((procrustes(rng.normal(size=(kt_l, ks_l))), procrustes(rng.normal(size=(kt_r, ks_r)))))
    best = None
    for P0, Q0 in inits:
        P, Q = _alt_procrustes_from(Cs, Ct, P0, Q0, iters)
        obj = _procrustes_objective(Cs, Ct, P, Q)
        if best is None or obj > best[0]:
            best = (obj, P, Q)
    _, P, Q = best
    pred = [P @ cs @ Q.T for cs in Cs]
    denom = sum(float(np.sum(p * p)) for p in pred)
    c = sum(float(np.sum(p * ct)) for p, ct in zip(pred, Ct)) / denom if denom > 0 else 0.0
    return P, Q, c


def _alt_bilinear(Cs, Ct, P, Q, ridge: float, iters: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(iters):
        X = [cs @ Q.T for cs in Cs]
        P = sum(ct @ x.T for ct, x in zip(Ct, X)) @ np.linalg.inv(sum(x @ x.T for x in X) + ridge * np.eye(X[0].shape[0]))
        Y = [cs.T @ P.T for cs in Cs]
        Q = sum(ct.T @ y.T for ct, y in zip(Ct, Y)) @ np.linalg.inv(sum(y @ y.T for y in Y) + ridge * np.eye(Y[0].shape[0]))
    return P, Q


def _fit_procrustes_full(src: list[LowRank], tgt: list[LowRank], ns: float, nt: float, iters: int,
                         k: int, ridge: float):
    # Initialise from the SVD-coordinate Procrustes solution lifted to full dimensions
    # (alternation from the identity gets stuck in poor local optima).
    Us, Vs, Ut, Vt, Cs, Ct = _fit_svd_coords(src, tgt, ns, nt, k)
    P, Q, _ = _alt_procrustes(Cs, Ct, 50, ridge)
    R_out = span_procrustes(Ut @ P, Us)
    R_in = span_procrustes(Vt @ Q, Vs)
    for _ in range(iters):
        # R_out = procrustes(Σ ΔW_t R_in ΔW_sᵀ); factors kept low rank.
        Ls, Rs = [], []
        for s, t in zip(src, tgt):
            G = (t.V.T @ R_in.apply(s.V)) * t.S[:, None] * s.S[None, :] / (ns * nt)
            Ls.append(t.U @ G)
            Rs.append(s.U)
        R_out = span_procrustes(np.concatenate(Ls, 1), np.concatenate(Rs, 1))
        Ls, Rs = [], []
        for s, t in zip(src, tgt):
            G = (t.U.T @ R_out.apply(s.U)) * t.S[:, None] * s.S[None, :] / (ns * nt)
            Ls.append(t.V @ G)
            Rs.append(s.V)
        R_in = span_procrustes(np.concatenate(Ls, 1), np.concatenate(Rs, 1))
    num = den = 0.0
    for s, t in zip(src, tgt):
        p = LowRank(R_out.apply(s.U), s.S / ns, R_in.apply(s.V))
        num += float(np.sum(((p.U.T @ t.U) * p.S[:, None]) * ((p.V.T @ t.V) * t.S[None, :]))) / nt
        den += p.fro() ** 2
    return R_out, R_in, (num / den if den > 0 else 0.0)


def fit_module(method: str, src: list[LowRank], tgt: list[LowRank], s_info: ModuleInfo, t_info: ModuleInfo,
               settings: MapSettings, base_svd: dict[str, Any] | None = None) -> ModuleMap:
    ns, nt = _norm(src), _norm(tgt)
    mm = ModuleMap(s_info.name, t_info.name, method, stats={"source_norm": ns, "target_norm": nt})
    same_shape = s_info.shape == t_info.shape
    if method == "identity":
        if not same_shape:
            mm.applicable, mm.note = False, f"shape {s_info.shape} != {t_info.shape}; naive copy undefined"
        return mm
    if method == "random":
        return mm
    if method == "cross_lora":
        if base_svd is None:
            mm.applicable, mm.note = False, "base-weight SVDs not provided"
            return mm
        mm.params = dict(base_svd)
        return mm
    if method == "procrustes_full":
        if not same_shape:
            mm.applicable, mm.note = False, "procrustes_full needs identical module shapes"
            return mm
        R_out, R_in, c = _fit_procrustes_full(src, tgt, ns, nt, max(1, settings.als_iters // 5),
                                              settings.k, settings.ridge)
        mm.params = {"Wo": R_out.W, "Ro": R_out.R, "Wi": R_in.W, "Ri": R_in.R}
        mm.stats["scale"] = c
        return mm
    Us, Vs, Ut, Vt, Cs, Ct = _fit_svd_coords(src, tgt, ns, nt, settings.k)
    P, Q, c = _alt_procrustes(Cs, Ct, settings.als_iters, settings.ridge)
    if method == "svd_procrustes":
        P = c * P
    elif method == "svd_linear":
        P, Q = _alt_bilinear(Cs, Ct, c * P, Q, settings.ridge, settings.als_iters)
    else:
        raise ValueError(f"unknown method {method}")
    mm.params = {"Us": Us, "Vs": Vs, "Ut": Ut, "Vt": Vt, "P": P, "Q": Q}
    fitted = [P @ cs @ Q.T for cs in Cs]
    mm.stats["train_coord_rel_err"] = float(np.sqrt(sum(np.sum((f - ct) ** 2) for f, ct in zip(fitted, Ct)) /
                                                    max(sum(np.sum(ct ** 2) for ct in Ct), 1e-30)))
    mm.stats["k_source"], mm.stats["k_target"] = Us.shape[1], Ut.shape[1]
    return mm


def predict_module(mm: ModuleMap, x: LowRank, t_shape: tuple[int, int], rng: np.random.Generator) -> LowRank | None:
    if not mm.applicable:
        return None
    ns, nt = mm.stats["source_norm"], mm.stats["target_norm"]
    if mm.method == "identity":
        return x
    if mm.method == "random":
        r = max(x.rank, 1)
        U = np.linalg.qr(rng.normal(size=(t_shape[0], r)))[0]
        V = np.linalg.qr(rng.normal(size=(t_shape[1], r)))[0]
        S = np.abs(rng.normal(size=r))
        target_norm = nt * x.fro() / ns
        return LowRank(U, S * target_norm / np.linalg.norm(S), V)
    if mm.method == "cross_lora":
        p = mm.params
        core = x.project(p["Us"], p["Vs"])  # source base-weight singular coordinates
        return from_factors(p["Ut"] @ core, p["Vt"].T)
    if mm.method == "procrustes_full":
        R_out = SpanRotation(mm.params["Wo"], mm.params["Ro"])
        R_in = SpanRotation(mm.params["Wi"], mm.params["Ri"])
        return LowRank(R_out.apply(x.U), x.S * mm.stats["scale"] * nt / ns, R_in.apply(x.V))
    p = mm.params
    ct = p["P"] @ (x.project(p["Us"], p["Vs"]) / ns) @ p["Q"].T
    return from_factors(p["Ut"] @ ct * nt, p["Vt"].T)


# ---------------------------------------------------------------------------
# Pair maps
# ---------------------------------------------------------------------------


@dataclass
class PairMap:
    method: str
    source_base: str
    target_base: str
    split: TaskSplit
    modules: dict[str, ModuleMap]
    correspondence: dict[str, Any]
    target_shapes: dict[str, tuple[int, int]]
    target_info: dict[str, dict[str, Any]]
    settings: dict[str, Any]
    seed: int = 0

    @property
    def map_id(self) -> str:
        return f"{self.source_base}__to__{self.target_base}__{self.split.split_id}__{self.method}"

    @property
    def applicable(self) -> bool:
        return any(m.applicable for m in self.modules.values())

    def predict(self, source: DeltaSet) -> DeltaSet:
        task = source.meta["task_id"]
        rng = np.random.default_rng([self.seed, int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)])
        out: dict[str, LowRank] = {}
        skipped = []
        for tname, mm in self.modules.items():
            if mm.source not in source.modules:
                skipped.append(tname)
                continue
            pred = predict_module(mm, source.modules[mm.source], self.target_shapes[tname], rng)
            if pred is None:
                skipped.append(tname)
            else:
                out[tname] = pred
        meta = {"kind": "predicted", "method": self.method, "map_id": self.map_id, "task_id": task,
                "source_base": self.source_base, "target_base": self.target_base,
                "source_run": source.meta.get("run_id"), "seed": source.meta.get("seed"),
                "split_id": self.split.split_id, "train_task_ids": list(self.split.train),
                "holdout_task_ids": list(self.split.holdout),
                "in_sample": task in self.split.train, "skipped_modules": skipped}
        return DeltaSet(meta, out, {n: self.target_info[n] for n in out})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {f"{t}::{k}": np.ascontiguousarray(v, dtype=np.float32)
                   for t, mm in self.modules.items() for k, v in mm.params.items()}
        if tensors:
            save_file(tensors, str(path / "map_params.safetensors"))
        meta = {"method": self.method, "source_base": self.source_base, "target_base": self.target_base,
                "split": asdict(self.split), "correspondence": self.correspondence,
                "target_shapes": self.target_shapes, "target_info": self.target_info,
                "settings": self.settings, "seed": self.seed, "map_id": self.map_id,
                "train_task_ids": list(self.split.train), "holdout_task_ids": list(self.split.holdout),
                "modules": {t: {"source": m.source, "method": m.method, "stats": m.stats,
                                "applicable": m.applicable, "note": m.note} for t, m in self.modules.items()}}
        (path / "map_meta.json").write_text(json.dumps(meta, indent=1))

    @staticmethod
    def load(path: str | Path) -> "PairMap":
        path = Path(path)
        meta = json.loads((path / "map_meta.json").read_text())
        tensors = load_file(str(path / "map_params.safetensors")) if (path / "map_params.safetensors").exists() else {}
        mods = {}
        for t, m in meta["modules"].items():
            params = {k.split("::", 1)[1]: v.astype(np.float64) for k, v in tensors.items() if k.split("::", 1)[0] == t}
            mods[t] = ModuleMap(m["source"], t, m["method"], params, m["stats"], m["applicable"], m["note"])
        s = meta["split"]
        return PairMap(meta["method"], meta["source_base"], meta["target_base"],
                       TaskSplit(s["split_id"], tuple(s["train"]), tuple(s["holdout"])), mods,
                       meta["correspondence"], {k: tuple(v) for k, v in meta["target_shapes"].items()},
                       meta["target_info"], meta["settings"], meta["seed"])


def fit_pair_map(method: str, source_deltas: dict[str, DeltaSet], target_deltas: dict[str, DeltaSet],
                 split: TaskSplit, settings: MapSettings, source_base: str, target_base: str,
                 base_svds: dict[str, dict[str, np.ndarray]] | None = None, seed: int = 0) -> PairMap:
    """Fit one source→target map from training-task updates only.

    ``source_deltas`` / ``target_deltas`` map task_id -> DeltaSet (one seed). Passing any
    held-out or undeclared task raises :class:`HoldoutLeakError`: the check is at the API
    boundary so no downstream code path can see held-out updates.
    """
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method}; known {ALL_METHODS}")
    split.check_fit_inputs(source_deltas.keys(), "source")
    split.check_fit_inputs(target_deltas.keys(), "target")
    tasks = [t for t in split.train if t in source_deltas and t in target_deltas]
    if not tasks:
        raise ValueError("no training task has both source and target updates")
    any_s, any_t = source_deltas[tasks[0]], target_deltas[tasks[0]]
    corr = match_modules(inventory_from_deltaset(any_s), inventory_from_deltaset(any_t), settings.layer_matching)
    modules: dict[str, ModuleMap] = {}
    for s_info, t_info in corr.pairs:
        src = [source_deltas[t].modules[s_info.name] for t in tasks]
        tgt = [target_deltas[t].modules[t_info.name] for t in tasks]
        bs = (base_svds or {}).get(f"{s_info.name}|{t_info.name}")
        modules[t_info.name] = fit_module(method, src, tgt, s_info, t_info, settings, bs)
    return PairMap(method, source_base, target_base, split, modules, corr.to_dict(),
                   {n: lr.shape for n, lr in any_t.modules.items()}, dict(any_t.info),
                   {**asdict(settings), "fit_tasks": tasks}, seed)


def base_weight_svds(source_model, target_model, corr: ModuleCorrespondence, k: int) -> dict[str, dict[str, np.ndarray]]:
    """Top-k singular vectors of matched base weights (for the data-free cross_lora baseline)."""
    import torch

    def topk(model, name):
        w = model.get_submodule(name).weight.detach().float().cpu()
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        u = u[:, :k].numpy().astype(np.float64)
        v = vh[:k].T.numpy().astype(np.float64)
        signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
        signs[signs == 0] = 1
        return u * signs, v * signs  # sign-fix each (u_i, v_i) pair jointly

    out = {}
    for s_info, t_info in corr.pairs:
        Us, Vs = topk(source_model, s_info.name)
        Ut, Vt = topk(target_model, t_info.name)
        kk = min(Us.shape[1], Ut.shape[1], Vs.shape[1], Vt.shape[1])
        out[f"{s_info.name}|{t_info.name}"] = {"Us": Us[:, :kk], "Vs": Vs[:, :kk], "Ut": Ut[:, :kk], "Vt": Vt[:, :kk]}
    return out


def reconstruction_report(pred: DeltaSet, true: DeltaSet) -> dict[str, Any]:
    """Diagnostic Δ reconstruction error of a prediction against a real target update."""
    from .spectral import cosine, relative_error

    per = {n: {"rel_err": relative_error(p, true.modules[n]), "cosine": cosine(p, true.modules[n])}
           for n, p in pred.modules.items() if n in true.modules}
    if not per:
        return {"n_modules": 0}
    return {"n_modules": len(per), "mean_rel_err": float(np.mean([v["rel_err"] for v in per.values()])),
            "mean_cosine": float(np.mean([v["cosine"] for v in per.values()])),
            "pred_norm": pred.fro(), "true_norm": true.fro(), "modules": per}
