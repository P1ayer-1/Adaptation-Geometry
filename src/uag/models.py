"""Loading frozen bases with provenance (exact revision / weight hash)."""

from __future__ import annotations

from typing import Any

import torch

from .config import BaseConfig, resolve_path
from .provenance import sha256_dir

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def base_location(base: BaseConfig) -> tuple[str, str | None]:
    """(path-or-hub-id, revision) to pass to ``from_pretrained``."""
    if base.source == "hub":
        return base.model_id, base.revision
    return str(resolve_path(base.model_id)), None


def ensure_base(base: BaseConfig) -> None:
    if base.source == "tiny":
        from .tiny import build_tiny_base

        build_tiny_base(base.tiny_spec or {}, resolve_path(base.model_id))


def load_tokenizer(base: BaseConfig):
    from transformers import AutoTokenizer

    ensure_base(base)
    loc, rev = base_location(base)
    tok_id = base.tokenizer_id or loc
    tok = AutoTokenizer.from_pretrained(tok_id, revision=base.tokenizer_revision or rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(base: BaseConfig, dtype: str | None = None, device: str = "auto"):
    """Load a base with every parameter frozen; returns (model, tokenizer, provenance)."""
    from transformers import AutoModelForCausalLM

    ensure_base(base)
    loc, rev = base_location(base)
    torch_dtype = DTYPES[dtype or base.dtype]
    model = AutoModelForCausalLM.from_pretrained(loc, revision=rev, dtype=torch_dtype)
    model.to(pick_device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = load_tokenizer(base)
    prov: dict[str, Any] = {
        "base_name": base.name,
        "base_model": base.model_id,
        "base_revision": base.revision,
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": base.tokenizer_revision or base.revision,
        "family": base.family,
        "model_type": model.config.model_type,
        "dtype": str(torch_dtype).replace("torch.", ""),
    }
    if base.source != "hub":
        prov["weights_sha256"] = sha256_dir(loc, "*.safetensors")
    return model, tok, prov


def state_fingerprint(model) -> str:
    """Cheap fingerprint of a model's frozen weights (to check they never change)."""
    import hashlib

    h = hashlib.sha256()
    def canonical(name: str) -> str:  # undo PEFT wrapping so wrapped/unwrapped models compare
        return name.removeprefix("base_model.model.").replace(".base_layer.", ".")

    for name, p in sorted((canonical(k), v) for k, v in model.state_dict().items()):
        if "lora_" in name:
            continue
        t = p.detach().float().cpu()
        h.update(name.encode())
        h.update(f"{t.sum().item():.8e}|{t.abs().sum().item():.8e}|{t.flatten()[:16].tolist()}".encode())
    return h.hexdigest()
