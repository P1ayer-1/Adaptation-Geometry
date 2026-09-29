"""Reconstruct effective LoRA updates ΔW = scale · B A and verify them against PEFT.

Raw A/B factors are never compared across runs, because the factorisation is non-unique
(BA = (BR)(R⁻¹A), spec §18). Instead every update is stored in its exact thin-SVD form
``ΔW = U diag(S) Vᵀ`` (rank ≤ r), which is compact and gauge-invariant up to signs of
singular-vector pairs. Dense matrices are only materialised module by module, on demand.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors.numpy import load_file as np_load_file
from safetensors.numpy import save_file as np_save_file
from safetensors.torch import load_file

from .alignment import classify_module, parse_layer, strip_peft_prefix


@dataclass
class LowRank:
    """Thin SVD ΔW = U diag(S) Vᵀ with U: (out, r), S: (r,), V: (in, r)."""

    U: np.ndarray
    S: np.ndarray
    V: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.U.shape[0], self.V.shape[0])

    @property
    def rank(self) -> int:
        return int(self.S.shape[0])

    def dense(self) -> np.ndarray:
        return (self.U * self.S) @ self.V.T

    def fro(self) -> float:
        return float(np.sqrt(np.sum(self.S.astype(np.float64) ** 2)))

    def scaled(self, c: float) -> "LowRank":
        if c >= 0:
            return LowRank(self.U, self.S * c, self.V)
        return LowRank(-self.U, self.S * -c, self.V)

    def truncate(self, k: int) -> "LowRank":
        return LowRank(self.U[:, :k], self.S[:k], self.V[:, :k])

    def __add__(self, other: "LowRank") -> "LowRank":
        if self.shape != other.shape:
            raise ValueError(f"shape mismatch {self.shape} vs {other.shape}")
        return from_factors(np.concatenate([self.U * self.S, other.U * other.S], 1),
                            np.concatenate([self.V, other.V], 1).T)

    def project(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Coordinates leftᵀ ΔW right without materialising ΔW."""
        return ((left.T @ self.U) * self.S) @ (self.V.T @ right)

    @staticmethod
    def from_dense(m: np.ndarray, rank: int | None = None, tol: float = 1e-10) -> "LowRank":
        u, s, vt = np.linalg.svd(np.asarray(m, dtype=np.float64), full_matrices=False)
        keep = s > tol * max(s.max(initial=0.0), 1e-30)
        if rank is not None:
            keep[rank:] = False
        return LowRank(u[:, keep], s[keep], vt[keep].T)


def from_factors(B: np.ndarray, A: np.ndarray, scale: float = 1.0, tol: float = 1e-10) -> LowRank:
    """Exact thin SVD of scale · B @ A via two QR factorisations (never forms the dense matrix)."""
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    qb, rb = np.linalg.qr(B)
    qa, ra = np.linalg.qr(A.T)
    u, s, vt = np.linalg.svd(scale * rb @ ra.T, full_matrices=False)
    keep = s > tol * max(s.max(initial=0.0), 1e-30)
    return LowRank(qb @ u[:, keep], s[keep], qa @ vt[keep].T)


@dataclass
class DeltaSet:
    """All effective updates of one adapter (or one predicted adapter)."""

    meta: dict[str, Any]
    modules: dict[str, LowRank]
    info: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {layer, cls}

    @property
    def task_id(self) -> str:
        return self.meta["task_id"]

    def fro(self) -> float:
        return float(np.sqrt(sum(m.fro() ** 2 for m in self.modules.values())))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for name, lr in self.modules.items():
            tensors[f"{name}::U"] = np.ascontiguousarray(lr.U, dtype=np.float32)
            tensors[f"{name}::S"] = np.ascontiguousarray(lr.S, dtype=np.float32)
            tensors[f"{name}::V"] = np.ascontiguousarray(lr.V, dtype=np.float32)
        np_save_file(tensors, str(path / "delta_factors.safetensors"))
        (path / "delta_meta.json").write_text(json.dumps({"meta": self.meta, "info": self.info}, indent=1))

    @staticmethod
    def load(path: str | Path) -> "DeltaSet":
        path = Path(path)
        tensors = np_load_file(str(path / "delta_factors.safetensors"))
        meta = json.loads((path / "delta_meta.json").read_text())
        names = sorted({k.rsplit("::", 1)[0] for k in tensors})
        mods = {n: LowRank(tensors[f"{n}::U"].astype(np.float64), tensors[f"{n}::S"].astype(np.float64),
                           tensors[f"{n}::V"].astype(np.float64)) for n in names}
        return DeltaSet(meta["meta"], mods, meta["info"])


def adapter_scaling(cfg: dict[str, Any], module_name: str) -> float:
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise NotImplementedError("rank_pattern/alpha_pattern adapters are not supported")
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r


def extract_deltas(adapter_dir: str | Path, model_type: str, meta: dict[str, Any] | None = None) -> DeltaSet:
    """Read a PEFT adapter and return every module's exact ΔW in factored SVD form."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        raise NotImplementedError("DoRA updates are not of the form scale·BA")
    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    a_keys = sorted(k for k in weights if ".lora_A." in k)
    modules, info = {}, {}
    for ka in a_keys:
        kb = ka.replace(".lora_A.", ".lora_B.")
        name = strip_peft_prefix(ka.split(".lora_A.")[0])
        A = weights[ka].float().numpy()
        B = weights[kb].float().numpy()
        if cfg.get("fan_in_fan_out"):
            raise NotImplementedError("fan_in_fan_out (Conv1D) adapters are not supported")
        modules[name] = from_factors(B, A, adapter_scaling(cfg, name))
        info[name] = {"layer": parse_layer(name), "cls": classify_module(name, model_type)}
    base_meta = {"kind": "lora", "adapter_dir": str(adapter_dir), "model_type": model_type,
                 "lora_rank": cfg["r"], "lora_alpha": cfg["lora_alpha"],
                 "use_rslora": bool(cfg.get("use_rslora"))}
    return DeltaSet({**base_meta, **(meta or {})}, modules, info)


def iter_dense(ds: DeltaSet) -> Iterator[tuple[str, np.ndarray]]:
    """Stream dense ΔW module by module (never holds more than one in memory)."""
    for name, lr in ds.modules.items():
        yield name, lr.dense()


# ---------------------------------------------------------------------------
# Applying updates to a base model
# ---------------------------------------------------------------------------


def _linear(model, name: str):
    root = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model, "peft_config"):
        root = model.base_model.model
    mod = root.get_submodule(name)
    return getattr(mod, "base_layer", mod)


@contextlib.contextmanager
def applied_delta(model, ds: DeltaSet, coef: float = 1.0):
    """Temporarily add coef · ΔW to the base weights (restored exactly on exit)."""
    saved = {}
    try:
        with torch.no_grad():
            for name, lr in ds.modules.items():
                lin = _linear(model, name)
                if tuple(lin.weight.shape) != lr.shape:
                    raise ValueError(f"{name}: delta shape {lr.shape} != weight {tuple(lin.weight.shape)}")
                saved[name] = lin.weight.detach().clone()
                d = torch.from_numpy(lr.dense()).to(device=lin.weight.device, dtype=torch.float32)
                lin.weight.copy_((lin.weight.float() + coef * d).to(lin.weight.dtype))
        yield model
    finally:
        with torch.no_grad():
            for name, w in saved.items():
                _linear(model, name).weight.copy_(w)


def verify_delta_reconstruction(peft_model, ds: DeltaSet, batch: dict[str, torch.Tensor],
                                atol: float = 1e-4, rtol: float = 1e-4) -> dict[str, Any]:
    """Acceptance test (spec §20.3): base + ΔW reproduces the active LoRA's logits.

    Also checks each reconstructed ΔW against PEFT's own ``get_delta_weight``. Run it on a
    float32 copy of the model: in bf16, adding ΔW into rounded weights differs from the
    unmerged LoRA path by rounding noise alone. Tolerance: atol + rtol · max|logit|.
    """
    per_module_err = 0.0
    root = peft_model.base_model.model
    for name, lr in ds.modules.items():
        lora_layer = root.get_submodule(name)
        ref = lora_layer.get_delta_weight("default").detach().float().cpu().numpy()
        per_module_err = max(per_module_err, float(np.abs(ref - lr.dense()).max()))
    peft_model.eval()
    with torch.no_grad():
        logits_lora = peft_model(**batch).logits.float()
        with peft_model.disable_adapter():
            logits_base = peft_model(**batch).logits.float()
            with applied_delta(peft_model, ds):
                logits_merged = peft_model(**batch).logits.float()
    max_diff = float((logits_lora - logits_merged).abs().max())
    adapter_effect = float((logits_lora - logits_base).abs().max())
    tol = atol + rtol * float(logits_lora.abs().max())
    return {"max_abs_logit_diff": max_diff, "adapter_effect_max_abs": adapter_effect,
            "max_abs_module_delta_err": per_module_err, "tolerance": tol,
            "passed": bool(max_diff <= tol and per_module_err <= atol),
            "n_modules": len(ds.modules)}
