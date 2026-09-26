"""Randomly initialised miniature bases + a byte-level tokenizer for smoke tests and CI.

They let the full pipeline (train -> extract -> evaluate -> map -> analyse) run on a CPU
without hub access. Results on tiny bases validate plumbing only; they carry no scientific
weight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>"]


def byte_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {c: i for i, c in enumerate(alphabet)}
    for s in SPECIAL_TOKENS:
        vocab[s] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIAL_TOKENS)
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>",
                                   eos_token="</s>")


def tiny_config(spec: dict[str, Any], tokenizer):
    from transformers import AutoConfig

    spec = dict(spec)
    model_type = spec.pop("model_type")
    spec.pop("init_seed", None)
    return AutoConfig.for_model(
        model_type,
        vocab_size=len(tokenizer),
        max_position_embeddings=spec.pop("max_position_embeddings", 1024),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
        **spec,
    )


def build_tiny_base(spec: dict[str, Any], out_dir: str | Path, overwrite: bool = False) -> Path:
    """Create (once) a tiny random base at ``out_dir``; deterministic given ``init_seed``."""
    from transformers import AutoModelForCausalLM

    out = Path(out_dir)
    if (out / "config.json").exists() and not overwrite:
        return out
    tok = byte_tokenizer()
    cfg = tiny_config(spec, tok)
    torch.manual_seed(int(spec.get("init_seed", 0)))
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    return out
