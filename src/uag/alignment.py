"""Semantic module matching across architectures and orthogonal alignment primitives.

Module matching is semantic and explicit (spec §13): every architecture maps its projection
names onto canonical classes ``q, k, v, o, up, down, gate``. A class that an architecture lacks
(e.g. no gate projection) or only has in fused form (e.g. ``qkv_proj``) is *recorded as an
omission*; nothing is ever silently reshaped or split.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from .config import CANONICAL_MODULES

_LLAMA_LIKE = {
    "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
    "o": "self_attn.o_proj", "up": "mlp.up_proj", "down": "mlp.down_proj", "gate": "mlp.gate_proj",
}

# model_type -> canonical class -> module-name suffix (None = absent in this architecture)
ARCH_MODULES: dict[str, dict[str, str | None]] = {
    **{mt: dict(_LLAMA_LIKE) for mt in (
        "llama", "mistral", "mixtral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "gemma3_text",
        "olmo", "olmo2", "granite", "stablelm", "smollm3", "cohere", "internlm2")},
    "phi": {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
            "o": "self_attn.dense", "up": "mlp.fc1", "down": "mlp.fc2", "gate": None},
}

# Architectures whose projections are fused; the fused classes are unsupported (omitted).
FUSED: dict[str, dict[str, str]] = {
    "phi3": {"q": "self_attn.qkv_proj", "k": "self_attn.qkv_proj", "v": "self_attn.qkv_proj",
             "up": "mlp.gate_up_proj", "gate": "mlp.gate_up_proj"},
}
ARCH_MODULES["phi3"] = {"q": None, "k": None, "v": None, "o": "self_attn.o_proj",
                        "up": None, "down": "mlp.down_proj", "gate": None}

_LAYER_RE = re.compile(r"\.(?:layers|h|blocks)\.(\d+)\.")


class ModuleMatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModuleInfo:
    name: str  # full module path in the base model, e.g. model.layers.3.self_attn.q_proj
    layer: int
    cls: str  # canonical class
    out_features: int
    in_features: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features)

    @property
    def slot(self) -> str:
        return f"L{self.layer}.{self.cls}"


@dataclass
class ModuleInventory:
    model_type: str
    num_layers: int
    modules: list[ModuleInfo]
    omissions: dict[str, str] = field(default_factory=dict)  # canonical class -> reason

    def by_slot(self) -> dict[tuple[int, str], ModuleInfo]:
        return {(m.layer, m.cls): m for m in self.modules}

    def names(self) -> list[str]:
        return [m.name for m in self.modules]

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": self.model_type, "num_layers": self.num_layers,
                "modules": [asdict(m) for m in self.modules], "omissions": self.omissions}


def parse_layer(name: str) -> int | None:
    m = _LAYER_RE.search("." + name + ".")
    return int(m.group(1)) if m else None


def arch_table(model_type: str) -> dict[str, str | None]:
    try:
        return ARCH_MODULES[model_type]
    except KeyError as e:
        raise ModuleMatchError(
            f"no semantic module table for model_type={model_type!r}; add it to ARCH_MODULES "
            f"explicitly rather than guessing") from e


def classify_module(name: str, model_type: str) -> str | None:
    """Canonical class of a module path, or None if it is not a mapped projection."""
    for cls, suffix in arch_table(model_type).items():
        if suffix and name.endswith("." + suffix):
            return cls
    return None


def discover_modules(model, wanted: Iterable[str] = CANONICAL_MODULES) -> ModuleInventory:
    """Enumerate the linear projections of ``model`` for the requested canonical classes."""
    import torch.nn as nn

    model_type = model.config.model_type
    table = arch_table(model_type)
    wanted = list(wanted)
    omissions: dict[str, str] = {}
    for cls in wanted:
        if table.get(cls) is None:
            fused = FUSED.get(model_type, {}).get(cls)
            omissions[cls] = (f"fused in {fused}; not split (never silently reshape)" if fused
                              else f"architecture {model_type} has no {cls} projection")
    modules = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        cls = classify_module(name, model_type)
        if cls is None or cls not in wanted:
            continue
        layer = parse_layer(name)
        if layer is None:
            raise ModuleMatchError(f"cannot parse layer index from {name}")
        modules.append(ModuleInfo(name, layer, cls, mod.out_features, mod.in_features))
    if not modules:
        raise ModuleMatchError(f"no target modules found in {model_type}")
    num_layers = int(getattr(model.config, "num_hidden_layers", max(m.layer for m in modules) + 1))
    modules.sort(key=lambda m: (m.layer, CANONICAL_MODULES.index(m.cls)))
    return ModuleInventory(model_type, num_layers, modules, omissions)


def strip_peft_prefix(name: str) -> str:
    """Map a PEFT-wrapped module path back to the base-model path."""
    for prefix in ("base_model.model.",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


# ---------------------------------------------------------------------------
# Cross-model correspondence
# ---------------------------------------------------------------------------


@dataclass
class ModuleCorrespondence:
    """For each target module, the source module it is predicted from.

    Layers are matched by relative depth: target layer j <- source layer
    round(j * (L_s - 1) / (L_t - 1)). The mapping is bijective only when both models
    have the same depth and the same set of canonical classes.
    """

    pairs: list[tuple[ModuleInfo, ModuleInfo]]  # (source, target)
    omissions: list[dict[str, Any]]
    bijective: bool

    def to_dict(self) -> dict[str, Any]:
        return {"pairs": [(s.name, t.name) for s, t in self.pairs], "omissions": self.omissions,
                "bijective": self.bijective}


def relative_depth_layer(j: int, n_target: int, n_source: int) -> int:
    if n_target == 1:
        return 0
    return int(round(j * (n_source - 1) / (n_target - 1)))


def match_modules(source: ModuleInventory, target: ModuleInventory,
                  layer_matching: str = "relative_depth") -> ModuleCorrespondence:
    if layer_matching != "relative_depth":
        raise ValueError(f"unknown layer_matching {layer_matching}")
    src = source.by_slot()
    pairs, omissions = [], []
    tgt_classes = {m.cls for m in target.modules}
    src_classes = {m.cls for m in source.modules}
    for cls in sorted(src_classes - tgt_classes):
        omissions.append({"class": cls, "side": "target", "reason": target.omissions.get(cls, "absent")})
    for cls in sorted(tgt_classes - src_classes):
        omissions.append({"class": cls, "side": "source", "reason": source.omissions.get(cls, "absent")})
    for t in target.modules:
        ls = relative_depth_layer(t.layer, target.num_layers, source.num_layers)
        s = src.get((ls, t.cls))
        if s is None:
            if t.cls in src_classes:
                omissions.append({"class": t.cls, "side": "source", "layer": ls, "reason": "missing layer"})
            continue
        pairs.append((s, t))
    used_sources = [s.name for s, _ in pairs]
    bijective = (len(set(used_sources)) == len(used_sources) == len(source.modules) == len(target.modules)
                 and not omissions)
    return ModuleCorrespondence(pairs, omissions, bijective)


def assert_bijective(corr: ModuleCorrespondence) -> None:
    if not corr.bijective:
        raise ModuleMatchError(f"module correspondence is not bijective; omissions={corr.omissions}")


# ---------------------------------------------------------------------------
# Orthogonal alignment primitives
# ---------------------------------------------------------------------------


def procrustes(m: np.ndarray) -> np.ndarray:
    """Orthogonal R maximising tr(R^T M), i.e. argmin ||R X - Y|| for M = Y X^T."""
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def orthogonal_procrustes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """R (orthogonal) minimising ||R x - y||_F for column-sample matrices x, y."""
    return procrustes(y @ x.T)


def canonical_sign(vectors: np.ndarray) -> np.ndarray:
    """Fix the sign ambiguity of singular vectors (columns): largest-|.| entry positive."""
    idx = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[idx, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1
    return vectors * signs
