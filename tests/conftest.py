import glob

import pytest

from uag.config import load_task


@pytest.fixture(scope="session")
def all_tasks():
    return [load_task(p) for p in sorted(glob.glob("configs/tasks/*.yaml"))]


@pytest.fixture(scope="session")
def v2_tasks():
    return [load_task(p) for p in sorted(glob.glob("configs/tasks/v2/*.yaml"))]


@pytest.fixture(scope="session")
def panel_tasks(all_tasks, v2_tasks):
    """Every committed dataset version (v1 panel + v2 panel)."""
    return all_tasks + v2_tasks


@pytest.fixture
def small_task(all_tasks, tmp_path):
    """A shrunken copy of T1 written to a temporary data dir."""
    import dataclasses

    t = dataclasses.replace(all_tasks[1], n_train=60, n_valid=20, n_test=20)
    return t, tmp_path / "data"


import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@pytest.fixture(scope="session")
def tiny_train_settings():
    from uag.config import TrainSettings

    return TrainSettings(learning_rate=3e-3, batch_size=4, max_tokens_seen=3000, max_seq_len=128,
                         eval_every_steps=5, valid_max_examples=8, dtype="float32", device="cpu")


@pytest.fixture(scope="session")
def tiny_bases():
    from uag.config import load_base

    return {n: load_base(f"configs/bases/{n}.yaml")
            for n in ("tiny_llama_a1", "tiny_llama_a2", "tiny_qwen2_b", "tiny_phi_c")}


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory, tiny_bases, all_tasks, tiny_train_settings):
    """One tiny rank-4 adapter shared by several tests (acceptance §20.2)."""
    from uag.config import LoraSettings
    from uag.train_lora import train_lora

    task = next(t for t in all_tasks if t.task_id == "T1_sentiment")
    runs = tmp_path_factory.mktemp("runs")
    run_dir = train_lora(tiny_bases["tiny_llama_a1"], task, LoraSettings(rank=4, alpha=8),
                         tiny_train_settings, seed=0, runs_dir=runs)
    return tiny_bases["tiny_llama_a1"], task, run_dir
