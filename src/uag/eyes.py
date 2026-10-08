"""Smart eyes: data-informed, task-free fixed LoRA A matrices ("eyes"), one set per base.

With ``train_A: false`` an adapter can only respond to the 16 fixed mixtures of its input that
A computes. Random mixtures see the dominant directions of a layer's input but blur faint ones;
in the dev runs frozen random A learned everything except T1, whose decisive feature (which of
40 product words occurs) is faint, and which trainable A solved by re-aiming A by up to 0.8-0.9
in a few layers. This module builds A from the layer's input statistics on generic text:

``whitened``  A = R · Σ^{-1/2}  (random rows after equalising every input direction's variance,
              so faint directions are seen as clearly as dominant ones)
``pca``       A = the top-r principal directions of the input (the dominant directions)

Σ is the uncentred second moment of the module's input over calibration tokens (generic text,
never task data), regularised by ``eps · mean eigenvalue``. Each A is rescaled so that its
output has the same mean energy on calibration data as PEFT's random init, E‖A x‖² = E‖A₀ x‖²,
keeping training dynamics comparable. Results are cached per base and recorded in run manifests.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from .alignment import ModuleInventory

INPUT_GROUP = {"q": "attn_in", "k": "attn_in", "v": "attn_in", "o": "o_in", "gate": "mlp_in", "up": "mlp_in",
               "down": "down_in"}


def calibration_texts(source: str, n_chars: int = 400_000) -> list[str]:
    """Generic, task-free calibration text. ``wikitext`` (WikiText-2 train, needs the hub) or
    ``builtin`` (deterministic synthetic sentences; offline fallback and tests)."""
    if source == "wikitext":
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        out, total = [], 0
        for t in ds["text"]:
            t = t.strip()
            if len(t) > 200 and not t.startswith("="):
                out.append(t)
                total += len(t)
                if total >= n_chars:
                    break
        return out
    if source == "builtin":
        from .tasks import CITIES, FIRST_NAMES, ITEMS, JOBS

        rng = random.Random("uag-eyes-builtin")
        verbs = ["visited", "bought", "described", "painted", "repaired", "studied", "sold", "found"]
        out = []
        for _ in range(max(1, n_chars // 300)):
            s = [f"{rng.choice(FIRST_NAMES)} the {rng.choice(JOBS)} {rng.choice(verbs)} {rng.randint(2, 90)} "
                 f"{rng.choice(ITEMS)}s in {rng.choice(CITIES)}." for _ in range(5)]
            out.append(" ".join(s))
        return out
    raise ValueError(f"unknown calibration source {source!r}")


@torch.no_grad()
def input_second_moments(model, tok, inventory: ModuleInventory, texts: list[str], max_tokens: int,
                         max_len: int = 256, batch_size: int = 8, layers_per_pass: int = 8) -> dict[tuple, torch.Tensor]:
    """Σ per (layer, input group), accumulated in fp32 over at most ``max_tokens`` real tokens.
    Processed a few layers per pass so the 8192-wide MLP inputs fit on small GPUs."""
    dev = next(model.parameters()).device
    # module names are paths inside the base model; a PEFT wrapper keeps it at base_model.model
    root = model.base_model.model if hasattr(model, "peft_config") else model
    enc = []
    total = 0
    for t in texts:
        ids = tok(t, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
        enc.append(ids)
        total += len(ids)
        if total >= max_tokens:
            break
    reps = {}
    for m in inventory.modules:
        reps.setdefault((m.layer, INPUT_GROUP[m.cls]), m)  # one module per shared input
    keys = sorted(reps)
    layers = sorted({k[0] for k in keys})
    out: dict[tuple, torch.Tensor] = {}
    state: dict[str, torch.Tensor] = {}  # current batch's padding mask, shared by all hooks
    for start in range(0, len(layers), layers_per_pass):
        chunk = set(layers[start:start + layers_per_pass])
        acc, handles = {}, []
        for key in keys:
            if key[0] not in chunk:
                continue
            mod = root.get_submodule(reps[key].name)
            acc[key] = torch.zeros(reps[key].in_features, reps[key].in_features, device=dev, dtype=torch.float32)

            def hook(_m, inp, _out, key=key):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                x = x[state["mask"].reshape(-1)]
                acc[key].addmm_(x.T, x)

            handles.append(mod.register_forward_hook(hook))
        try:
            for i in range(0, len(enc), batch_size):
                chunk_ids = enc[i:i + batch_size]
                L = max(len(s) for s in chunk_ids)
                ids = torch.full((len(chunk_ids), L), tok.pad_token_id or 0, dtype=torch.long)
                att = torch.zeros((len(chunk_ids), L), dtype=torch.long)
                for j, s in enumerate(chunk_ids):
                    ids[j, :len(s)] = torch.tensor(s)
                    att[j, :len(s)] = 1
                state["mask"] = att.bool().to(dev)
                model(input_ids=ids.to(dev), attention_mask=att.to(dev))
        finally:
            for h in handles:
                h.remove()
        n = float(sum(len(s) for s in enc))
        for key, a in acc.items():
            out[key] = (a / n).cpu()
    return out


def eig(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition in float64 (on the GPU when available), returned on the CPU."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    evals, evecs = torch.linalg.eigh(sigma.double().to(dev))
    return evals.clamp_min(0).cpu(), evecs.cpu()


def eye_matrix(sigma: torch.Tensor, a0: torch.Tensor, kind: str, gen: torch.Generator, eps: float = 1e-2,
               decomposition: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """Fixed A (r × d) of the given kind, scaled so that E‖A x‖² equals E‖A₀ x‖² under Σ."""
    r, d = a0.shape
    sigma = sigma.double()
    evals, evecs = decomposition or eig(sigma)
    if kind == "whitened":
        lam = eps * float(evals.mean())
        w = evecs @ torch.diag((evals + lam).rsqrt()) @ evecs.T
        a = torch.randn(r, d, generator=gen, dtype=torch.float64) @ w
    elif kind == "pca":
        a = evecs[:, -r:].flip(1).T.clone()
    else:
        raise ValueError(f"unknown eye kind {kind!r}")
    energy = lambda m: float(torch.trace(m @ sigma @ m.T))  # noqa: E731
    a = a * (energy(a0.double()) / max(energy(a), 1e-30)) ** 0.5
    return a.to(a0.dtype)


def build_eyes(model, tok, inventory: ModuleInventory, lora_a: dict[str, torch.Tensor], kind: str, source: str,
               max_tokens: int, seed: int) -> dict[str, torch.Tensor]:
    """A for every adapted module (keys = module names), from the current random inits ``lora_a``."""
    sigmas = input_second_moments(model, tok, inventory, calibration_texts(source), max_tokens)
    gen = torch.Generator().manual_seed(seed)
    decomp = {key: eig(sig) for key, sig in sigmas.items()}  # once per shared input (q/k/v, gate/up)
    out = {}
    for m in inventory.modules:
        key = (m.layer, INPUT_GROUP[m.cls])
        out[m.name] = eye_matrix(sigmas[key], lora_a[m.name].detach().float().cpu(), kind, gen,
                                 decomposition=decomp[key])
    return out


def eyes_cache_path(cache_dir: str | Path, base_name: str, kind: str, rank: int, source: str, max_tokens: int,
                    seed: int) -> Path:
    return Path(cache_dir) / f"{base_name}_{kind}_r{rank}_{source}_{max_tokens}_s{seed}.safetensors"


def apply_eyes(peft_model, tok, inventory: ModuleInventory, base_name: str, kind: str, source: str,
               max_tokens: int, seed: int, cache_dir: str | Path) -> dict[str, Any]:
    """Replace every lora_A weight with smart eyes (computed once per base, then cached)."""
    from safetensors.torch import load_file, save_file

    from .provenance import sha256_file

    root = peft_model.base_model.model
    lora_a = {m.name: root.get_submodule(m.name).lora_A["default"].weight for m in inventory.modules}
    path = eyes_cache_path(cache_dir, base_name, kind, next(iter(lora_a.values())).shape[0], source, max_tokens, seed)
    if path.exists():
        eyes = load_file(str(path))
    else:
        eyes = build_eyes(peft_model, tok, inventory, lora_a, kind, source, max_tokens, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in eyes.items()}, str(path))
    with torch.no_grad():
        for name, w in lora_a.items():
            w.copy_(eyes[name].to(device=w.device, dtype=w.dtype))
    return {"kind": kind, "calibration": source, "calibration_tokens": max_tokens, "cache": str(path),
            "sha256": sha256_file(path)}
