"""Stage 2: a shared task latent z_t with thin per-model decoders D_m (spec §7).

Only meant to be used once Stage 0/1 show task-general mapping. Every model m expresses
each module's update in its *own* truncated-SVD coordinate system (built from training
tasks only, as in :mod:`uag.transfer`); the latent must explain all models at once:

``linear``    vec(C_{m,·,t}) ≈ D_m z_t            (z_t ∈ R^{d_z}; one matrix per model)
``bilinear``  C_{m,j,t} ≈ P_{m,j} Z_t Q_{m,j}ᵀ     (Z_t ∈ R^{k_z×k_z}; far lower capacity)

Identifiability convention: latents are whitened over training tasks (mean 0 is *not*
imposed, second moment = I), so comparisons are meaningful up to an orthogonal rotation.

Protocols (random train/test splitting over base×task cells is deliberately unsupported):
:func:`leave_one_task_out` and :func:`leave_one_model_out`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .extract_delta import DeltaSet, LowRank, from_factors
from .transfer import HoldoutLeakError, _alt_procrustes, _coordinate_basis, _norm

Deltas = dict[str, dict[str, DeltaSet]]  # model -> task -> DeltaSet


@dataclass
class ModelCoords:
    """Per-model coordinate systems (bases + norms) fitted on training tasks only."""

    modules: list[str]
    bases: dict[str, tuple[np.ndarray, np.ndarray]]  # module -> (U_basis, V_basis)
    norms: dict[str, float]
    shapes: dict[str, tuple[int, int]]
    info: dict[str, dict[str, Any]]

    def encode(self, ds: DeltaSet) -> dict[str, np.ndarray]:
        return {n: ds.modules[n].project(*self.bases[n]) / self.norms[n] for n in self.modules}

    def vec(self, coords: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([coords[n].ravel() for n in self.modules])

    def unvec(self, v: np.ndarray) -> dict[str, np.ndarray]:
        out, i = {}, 0
        for n in self.modules:
            ku, kv = self.bases[n][0].shape[1], self.bases[n][1].shape[1]
            out[n] = v[i:i + ku * kv].reshape(ku, kv)
            i += ku * kv
        return out

    def decode(self, coords: dict[str, np.ndarray], meta: dict[str, Any]) -> DeltaSet:
        mods = {}
        for n, c in coords.items():
            U, V = self.bases[n]
            mods[n] = from_factors(U @ c * self.norms[n], V.T)
        return DeltaSet(meta, mods, {n: self.info[n] for n in mods})


def fit_coords(per_task: dict[str, DeltaSet], train_tasks: list[str], k: int) -> ModelCoords:
    leaked = set(per_task) - set(train_tasks)
    if leaked:
        raise HoldoutLeakError(f"coordinate systems may only see training tasks; got {sorted(leaked)}")
    tasks = [t for t in train_tasks if t in per_task]
    any_ds = per_task[tasks[0]]
    modules = sorted(any_ds.modules)
    bases, norms = {}, {}
    for n in modules:
        xs: list[LowRank] = [per_task[t].modules[n] for t in tasks]
        bases[n] = (_coordinate_basis(xs, k, "U"), _coordinate_basis(xs, k, "V"))
        norms[n] = _norm(xs)
    return ModelCoords(modules, bases, norms, {n: any_ds.modules[n].shape for n in modules}, dict(any_ds.info))


def _whiten(Z: np.ndarray) -> np.ndarray:
    """Return W such that (W Z)(W Z)ᵀ / T = I (Z: d_z × T)."""
    cov = Z @ Z.T / Z.shape[1]
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)
    return evecs @ np.diag(evals ** -0.5) @ evecs.T


@dataclass
class LatentModel:
    decoder: str
    d_z: int
    k: int
    ridge: float
    train_tasks: list[str]
    coords: dict[str, ModelCoords] = field(default_factory=dict)
    latents: dict[str, np.ndarray] = field(default_factory=dict)  # task -> z (vector or k_z×k_z)
    decoders: dict[str, Any] = field(default_factory=dict)  # model -> D (linear) | {module: (P, Q)}

    # -- encoding / decoding ----------------------------------------------------
    def _k_z(self) -> int:
        return int(round(np.sqrt(self.d_z)))

    def encode(self, observations: dict[str, DeltaSet]) -> np.ndarray:
        """Infer z for a task from any subset of models' updates (ridge least squares)."""
        rows, ys = [], []
        for m, ds in observations.items():
            c = self.coords[m].encode(ds)
            if self.decoder == "linear":
                rows.append(self.decoders[m])
                ys.append(self.coords[m].vec(c))
            else:
                for n, (P, Q) in self.decoders[m].items():
                    rows.append(np.kron(Q, P))  # vec(P Z Qᵀ) = (Q ⊗ P) vec(Z) (column-major vec)
                    ys.append(c[n].ravel(order="F"))
        A, y = np.concatenate(rows, 0), np.concatenate(ys)
        z = np.linalg.solve(A.T @ A + self.ridge * np.eye(A.shape[1]), A.T @ y)
        return z if self.decoder == "linear" else z.reshape(self._k_z(), self._k_z(), order="F")

    def decode_coords(self, model: str, z: np.ndarray) -> dict[str, np.ndarray]:
        if self.decoder == "linear":
            return self.coords[model].unvec(self.decoders[model] @ z)
        return {n: P @ z @ Q.T for n, (P, Q) in self.decoders[model].items()}

    def predict(self, model: str, z: np.ndarray, meta: dict[str, Any]) -> DeltaSet:
        return self.coords[model].decode(self.decode_coords(model, z), {"kind": "latent", "decoder": self.decoder,
                                                                         "target_base": model, **meta})

    # -- fitting -----------------------------------------------------------------
    def fit_decoder(self, model: str, per_task: dict[str, DeltaSet], tasks: list[str], iters: int = 30) -> None:
        """Fit (or refit) one model's decoder with the latents frozen."""
        coords = self.coords[model]
        C = {t: coords.encode(per_task[t]) for t in tasks}
        if self.decoder == "linear":
            Z = np.stack([self.latents[t] for t in tasks], 1)
            Y = np.stack([coords.vec(C[t]) for t in tasks], 1)
            self.decoders[model] = Y @ Z.T @ np.linalg.inv(Z @ Z.T + self.ridge * np.eye(Z.shape[0]))
            return
        dec = {}
        kz = self._k_z()
        prev = self.decoders.get(model, {})
        for n in coords.modules:
            ku, kv = coords.bases[n][0].shape[1], coords.bases[n][1].shape[1]
            if n in prev:
                P, Q = prev[n]  # warm start
            else:  # multi-start orthogonal solution, then refined by bilinear ALS (non-convex)
                P, Q, c = _alt_procrustes([self.latents[t] for t in tasks], [C[t][n] for t in tasks], 50, self.ridge)
                P = c * P
            for _ in range(iters):
                X = [self.latents[t] @ Q.T for t in tasks]
                P = sum(C[t][n] @ x.T for t, x in zip(tasks, X)) @ np.linalg.inv(
                    sum(x @ x.T for x in X) + self.ridge * np.eye(kz))
                Yb = [self.latents[t].T @ P.T for t in tasks]
                Q = sum(C[t][n].T @ y.T for t, y in zip(tasks, Yb)) @ np.linalg.inv(
                    sum(y @ y.T for y in Yb) + self.ridge * np.eye(kz))
            dec[n] = (P, Q)
        self.decoders[model] = dec

    def fit(self, deltas: Deltas, iters: int = 30, seed: int = 0) -> "LatentModel":
        for m, per_task in deltas.items():
            leaked = set(per_task) - set(self.train_tasks)
            if leaked:
                raise HoldoutLeakError(f"model {m}: non-training tasks {sorted(leaked)} passed to latent fitting")
            self.coords[m] = fit_coords(per_task, self.train_tasks, self.k)
        tasks = [t for t in self.train_tasks if any(t in d for d in deltas.values())]
        # Initialise latents from the SVD of the stacked (models × coordinates) × tasks matrix.
        rng = np.random.default_rng(seed)
        if self.decoder == "linear":
            stack = []
            for m, per_task in deltas.items():
                cm = self.coords[m]
                stack.append(np.stack([cm.vec(cm.encode(per_task[t])) if t in per_task
                                       else np.zeros(sum(b[0].shape[1] * b[1].shape[1] for b in cm.bases.values()))
                                       for t in tasks], 1))
            u, s, vt = np.linalg.svd(np.concatenate(stack, 0), full_matrices=False)
            d = min(self.d_z, vt.shape[0])
            Z = np.zeros((self.d_z, len(tasks)))
            Z[:d] = vt[:d] * s[:d, None]
            Z[d:] = 1e-3 * rng.normal(size=(self.d_z - d, len(tasks)))
            Z = _whiten(Z) @ Z
            self.latents = {t: Z[:, i] for i, t in enumerate(tasks)}
        else:
            kz = self._k_z()
            self.latents = {t: rng.normal(size=(kz, kz)) / np.sqrt(kz) for t in tasks}
        for _ in range(iters):
            for m, per_task in deltas.items():
                self.fit_decoder(m, per_task, [t for t in tasks if t in per_task], iters=3)
            for t in tasks:
                self.latents[t] = self.encode({m: d[t] for m, d in deltas.items() if t in d})
            if self.decoder == "linear":
                Z = np.stack([self.latents[t] for t in tasks], 1)
                W = _whiten(Z)
                self.latents = {t: W @ self.latents[t] for t in tasks}
                Winv = np.linalg.inv(W)
                self.decoders = {m: D @ Winv for m, D in self.decoders.items()}
        return self

    def train_reconstruction_error(self, deltas: Deltas) -> float:
        num = den = 0.0
        for m, per_task in deltas.items():
            for t, ds in per_task.items():
                if t not in self.latents:
                    continue
                c = self.coords[m].encode(ds)
                chat = self.decode_coords(m, self.latents[t])
                num += sum(float(np.sum((chat[n] - c[n]) ** 2)) for n in c)
                den += sum(float(np.sum(c[n] ** 2)) for n in c)
        return float(np.sqrt(num / den)) if den else float("nan")


def leave_one_task_out(deltas: Deltas, held_out: str, source_models: list[str], decoder: str = "linear",
                       d_z: int = 8, k: int = 16, ridge: float = 1e-2) -> dict[str, Any]:
    """Fit on every other task; infer z for ``held_out`` from ``source_models`` only; decode
    for every other model. Returns predictions (DeltaSets) keyed by target model."""
    tasks = sorted({t for d in deltas.values() for t in d} - {held_out})
    train = {m: {t: ds for t, ds in d.items() if t != held_out} for m, d in deltas.items()}
    lm = LatentModel(decoder, d_z, k, ridge, tasks).fit(train)
    z = lm.encode({m: deltas[m][held_out] for m in source_models})
    preds = {m: lm.predict(m, z, {"task_id": held_out, "protocol": "LOTO", "source_models": source_models})
             for m in deltas if m not in source_models}
    return {"model": lm, "z": z, "predictions": preds}


def leave_one_model_out(deltas: Deltas, new_model: str, fit_tasks: list[str], decoder: str = "linear",
                        d_z: int = 8, k: int = 16, ridge: float = 1e-2) -> dict[str, Any]:
    """Freeze {z_t, shared structure} learned without ``new_model``; fit only its decoder on
    ``fit_tasks``; predict its updates for every other task (unseen on the new model)."""
    others = {m: d for m, d in deltas.items() if m != new_model}
    tasks = sorted({t for d in others.values() for t in d})
    lm = LatentModel(decoder, d_z, k, ridge, tasks).fit(others)
    new_fit = {t: deltas[new_model][t] for t in fit_tasks}
    lm.coords[new_model] = fit_coords(new_fit, list(fit_tasks), k)
    lm.fit_decoder(new_model, new_fit, list(fit_tasks))
    test_tasks = [t for t in tasks if t not in fit_tasks]
    preds = {t: lm.predict(new_model, lm.latents[t], {"task_id": t, "protocol": "LOMO"}) for t in test_tasks}
    return {"model": lm, "predictions": preds, "test_tasks": test_tasks}
