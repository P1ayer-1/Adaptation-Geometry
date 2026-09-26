import glob

import pytest

from uag.config import load_task


@pytest.fixture(scope="session")
def all_tasks():
    return [load_task(p) for p in sorted(glob.glob("configs/tasks/*.yaml"))]


@pytest.fixture
def small_task(all_tasks, tmp_path):
    """A shrunken copy of T1 written to a temporary data dir."""
    import dataclasses

    t = dataclasses.replace(all_tasks[1], n_train=60, n_valid=20, n_test=20)
    return t, tmp_path / "data"
