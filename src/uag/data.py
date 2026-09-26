"""Canonical example schema, dataset generation, immutable manifests (spec §10).

Datasets live at ``data/generated/<task_id>/v<version>/<split>.jsonl``; their manifests at
``data/manifests/<task_id>.v<version>.yaml`` record SHA-256 hashes of every split. Loading
a split always re-verifies its hash, so a silently edited or regenerated dataset cannot be
used under an old version number.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import yaml

from . import __version__
from .config import REPO_ROOT, TaskConfig, dump_yaml
from .provenance import git_commit, sha256_file
from .tasks import get_generator

SPLITS = ("train", "valid", "test")
# Test is generated first; later splits exclude every input already used, so no evaluation
# input can ever appear in training data.
_GENERATION_ORDER = ("test", "valid", "train")
_SPLIT_SEED_OFFSET = {"test": 1, "valid": 2, "train": 3}

REQUIRED_KEYS = ("example_id", "task_id", "split", "input", "target", "metadata")
REQUIRED_META = ("source", "license", "generator_version", "difficulty")


class DataIntegrityError(RuntimeError):
    pass


def validate_example(ex: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in ex]
    if missing:
        raise DataIntegrityError(f"example missing keys {missing}: {ex.get('example_id')}")
    if ex["split"] not in SPLITS:
        raise DataIntegrityError(f"bad split {ex['split']!r}")
    missing_meta = [k for k in REQUIRED_META if k not in ex["metadata"]]
    if missing_meta:
        raise DataIntegrityError(f"{ex['example_id']}: metadata missing {missing_meta}")


def dataset_dir(task: TaskConfig, data_dir: str | Path = "data") -> Path:
    root = Path(data_dir) if Path(data_dir).is_absolute() else REPO_ROOT / data_dir
    return root / "generated" / task.task_id / f"v{task.version}"


def manifest_path(task: TaskConfig, data_dir: str | Path = "data") -> Path:
    root = Path(data_dir) if Path(data_dir).is_absolute() else REPO_ROOT / data_dir
    return root / "manifests" / f"{task.dataset_key}.yaml"


def instruction_for(task: TaskConfig) -> str:
    return get_generator(task.generator).instruction


def _generate_split(task: TaskConfig, split: str, n: int, exclude: set[str]) -> list[dict[str, Any]]:
    gen = get_generator(task.generator)
    rng = random.Random(task.seed * 1000 + _SPLIT_SEED_OFFSET[split])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts, max_attempts = 0, 200 * n
    while len(out) < n:
        attempts += 1
        if attempts > max_attempts:
            raise DataIntegrityError(
                f"{task.task_id}/{split}: could only generate {len(out)}/{n} unique inputs; "
                f"the generator lacks diversity for this size")
        ex = gen.fn(rng, **task.generator_kwargs)
        if ex["input"] in exclude or ex["input"] in seen:
            continue
        seen.add(ex["input"])
        meta = {"source": f"uag.tasks.{gen.name}", "license": task.license,
                "generator_version": gen.version, **ex["metadata"]}
        meta.setdefault("difficulty", "plain")
        out.append({"example_id": f"{task.task_id}-{split}-{len(out):06d}", "task_id": task.task_id,
                    "split": split, "input": ex["input"], "target": ex["target"], "metadata": meta})
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_dataset(task: TaskConfig, data_dir: str | Path = "data", overwrite: bool = False) -> dict[str, Any]:
    """Generate all splits and write the manifest.

    If a manifest for this ``task_id``/version already exists, the regenerated hashes must
    match it exactly; otherwise a :class:`DataIntegrityError` tells you to bump the version.
    """
    gen = get_generator(task.generator)
    if gen.version != str(task.generator_version):
        raise DataIntegrityError(f"{task.task_id}: config generator_version {task.generator_version} "
                                 f"!= code generator version {gen.version}")
    sizes = {"train": task.n_train, "valid": task.n_valid, "test": task.n_test}
    used: set[str] = set()
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in _GENERATION_ORDER:
        splits[split] = _generate_split(task, split, sizes[split], exclude=used)
        used.update(ex["input"] for ex in splits[split])

    ddir = dataset_dir(task, data_dir)
    for split in SPLITS:
        write_jsonl(ddir / f"{split}.jsonl", splits[split])
    hashes = {f"{s}_sha256": sha256_file(ddir / f"{s}.jsonl") for s in SPLITS}

    manifest = {
        "task_id": task.task_id,
        "name": task.name,
        "version": task.version,
        "task_class": task.task_class,
        **hashes,
        "n_train": task.n_train, "n_valid": task.n_valid, "n_test": task.n_test,
        "metric": task.metric,
        "primary_metric_direction": task.metric_direction,
        "scoring": task.scoring,
        "prompt_template_version": task.prompt_template_version,
        "instruction": gen.instruction,
        "generator": task.generator,
        "generator_version": gen.version,
        "generator_kwargs": task.generator_kwargs,
        "seed": task.seed,
        "license": task.license,
        "contamination_notes": task.contamination_notes,
        "split_disjointness": "inputs unique within and across splits (test generated first)",
        "uag_version": __version__,
        "code_commit": git_commit()["commit"],
    }
    mpath = manifest_path(task, data_dir)
    if mpath.exists() and not overwrite:
        old = yaml.safe_load(mpath.read_text())
        diffs = [k for k in hashes if old.get(k) != hashes[k]]
        if diffs:
            raise DataIntegrityError(
                f"{task.dataset_key}: regenerated data differs from the committed manifest ({diffs}). "
                f"Generator/template changes must create a new dataset version.")
        return old
    dump_yaml(manifest, mpath)
    return manifest


def load_manifest(task: TaskConfig, data_dir: str | Path = "data") -> dict[str, Any]:
    mpath = manifest_path(task, data_dir)
    if not mpath.exists():
        raise DataIntegrityError(f"no manifest for {task.dataset_key} at {mpath}; run `uag make-data`")
    return yaml.safe_load(mpath.read_text())


def load_split(task: TaskConfig, split: str, data_dir: str | Path = "data",
               verify: bool = True) -> list[dict[str, Any]]:
    """Load a split, verifying its SHA-256 against the immutable manifest."""
    if split not in SPLITS:
        raise ValueError(split)
    manifest = load_manifest(task, data_dir)
    path = dataset_dir(task, data_dir) / f"{split}.jsonl"
    if not path.exists():
        # Data are regenerable: rebuild deterministically and check against the manifest.
        generate_dataset(task, data_dir)
    if verify:
        actual = sha256_file(path)
        if actual != manifest[f"{split}_sha256"]:
            raise DataIntegrityError(f"{task.dataset_key}/{split}: hash {actual} != manifest "
                                     f"{manifest[f'{split}_sha256']}")
    rows = read_jsonl(path)
    for r in rows:
        validate_example(r)
        if r["split"] != split or r["task_id"] != task.task_id:
            raise DataIntegrityError(f"{r['example_id']}: wrong split/task")
    return rows


def dataset_hashes(task: TaskConfig, data_dir: str | Path = "data") -> dict[str, str]:
    m = load_manifest(task, data_dir)
    return {s: m[f"{s}_sha256"] for s in SPLITS}
