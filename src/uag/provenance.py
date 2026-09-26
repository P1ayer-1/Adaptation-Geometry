"""Provenance helpers: code commit, hashes, hardware and determinism settings (spec §21)."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .config import REPO_ROOT


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: str | Path, pattern: str = "*") -> str:
    """Hash of every file under ``path`` (sorted relative paths + contents)."""
    path = Path(path)
    h = hashlib.sha256()
    for p in sorted(q for q in path.rglob(pattern) if q.is_file()):
        h.update(str(p.relative_to(path)).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def git_commit() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    commit = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit or "unknown", "dirty": dirty}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(),
                            "cpu_count": os.cpu_count()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda"] = torch.version.cuda
    except ImportError:  # pragma: no cover
        pass
    for mod in ("transformers", "peft", "accelerate"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:  # pragma: no cover
            pass
    return info


def set_determinism(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed every RNG and request deterministic kernels; return what was actually set.

    Nondeterministic CUDA settings are recorded rather than silently assumed (spec §13).
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    record: dict[str, Any] = {"seed": seed, "deterministic_requested": deterministic}
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    record["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    record["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    record["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    record["note"] = ("Some CUDA kernels (e.g. scatter/index_add backward, flash attention) remain "
                      "nondeterministic even with these flags; warn_only=True keeps them running.")
    return record
