"""Map fitting: holdout guard (spec §13), recovery on synthetic shared geometry, controls."""

import numpy as np
import pytest

from uag.config import MapSettings
from uag.extract_delta import DeltaSet, LowRank
from uag.transfer import (HoldoutLeakError, PairMap, TaskSplit, fit_pair_map, reconstruction_report)

TASKS = [f"T{i}" for i in range(1, 11)]
SPLIT = TaskSplit.from_holdout("s0", TASKS, ["T9", "T10"])


def _orth(rng, d, k):
    return np.linalg.qr(rng.normal(size=(d, k)))[0]


def synthetic_world(seed=0, k=6, shared=True, same_shape=False):
    """Two 'models' whose per-module updates are D_m(z_t) = A_m Z_t B_mᵀ (shared latent Z_t)."""
    rng = np.random.default_rng(seed)
    dims = {"src": (24, 20), "tgt": (24, 20) if same_shape else (30, 16)}
    layers = 2
    Z = {t: rng.normal(size=(k, k)) for t in TASKS}
    worlds = {}
    for side, (do, di) in dims.items():
        A = {l: _orth(rng, do, k) for l in range(layers)}
        B = {l: _orth(rng, di, k) for l in range(layers)}
        per_task = {}
        for t in TASKS:
            z = Z[t] if (shared or side == "src") else rng.normal(size=(k, k))
            mods, info = {}, {}
            for l in range(layers):
                name = f"model.layers.{l}.self_attn.q_proj"
                mods[name] = LowRank.from_dense(A[l] @ z @ B[l].T * (3.0 if side == "tgt" else 1.0))
                info[name] = {"layer": l, "cls": "q"}
            per_task[t] = DeltaSet({"task_id": t, "model_type": "llama", "num_layers": layers}, mods, info)
        worlds[side] = per_task
    return worlds


def _fit(method, worlds, split=SPLIT, **kw):
    settings = MapSettings(k=kw.pop("k", 6), als_iters=60)
    src = {t: worlds["src"][t] for t in split.train}
    tgt = {t: worlds["tgt"][t] for t in split.train}
    return fit_pair_map(method, src, tgt, split, settings, "src", "tgt", **kw)


def test_holdout_ids_cannot_enter_fitting():
    w = synthetic_world()
    settings = MapSettings(k=6)
    leaky_tgt = {t: w["tgt"][t] for t in TASKS}  # includes T9/T10
    src = {t: w["src"][t] for t in SPLIT.train}
    with pytest.raises(HoldoutLeakError):
        fit_pair_map("svd_procrustes", src, leaky_tgt, SPLIT, settings, "src", "tgt")
    with pytest.raises(HoldoutLeakError):
        fit_pair_map("svd_procrustes", {**src, "T9": w["src"]["T9"]}, {t: w["tgt"][t] for t in SPLIT.train},
                     SPLIT, settings, "src", "tgt")
    with pytest.raises(HoldoutLeakError):
        TaskSplit("bad", ("T1", "T9"), ("T9",))
    m = _fit("svd_procrustes", w)
    assert set(m.split.holdout) == {"T9", "T10"} and "T9" not in m.settings["fit_tasks"]


@pytest.mark.parametrize("method", ["svd_procrustes", "svd_linear"])
def test_shared_geometry_is_recovered_on_heldout_tasks(method):
    w = synthetic_world(shared=True)
    m = _fit(method, w)
    for t in SPLIT.holdout:
        pred = m.predict(w["src"][t])
        assert not pred.meta["in_sample"]
        rep = reconstruction_report(pred, w["tgt"][t])
        assert rep["mean_rel_err"] < 0.05, (method, rep["mean_rel_err"])


def test_unrelated_geometry_does_not_transfer():
    w = synthetic_world(shared=False)
    m = _fit("svd_procrustes", w)
    rep = reconstruction_report(m.predict(w["src"]["T9"]), w["tgt"]["T9"])
    assert rep["mean_rel_err"] > 0.7


def test_procrustes_full_and_identity_same_shape():
    w = synthetic_world(shared=True, same_shape=True)
    m = _fit("procrustes_full", w)
    rep = reconstruction_report(m.predict(w["src"]["T9"]), w["tgt"]["T9"])
    assert rep["mean_rel_err"] < 0.05
    ident = _fit("identity", w)
    assert ident.applicable
    other = _fit("identity", synthetic_world(same_shape=False))
    assert not other.applicable and all(not mm.applicable for mm in other.modules.values())
    assert other.predict(w["src"]["T9"]).modules == {}


def test_random_baseline_is_norm_matched():
    w = synthetic_world()
    m = _fit("random", w)
    src = w["src"]["T9"]
    pred = m.predict(src)
    for n, lr in pred.modules.items():
        mm = m.modules[n]
        expect = mm.stats["target_norm"] * src.modules[mm.source].fro() / mm.stats["source_norm"]
        assert lr.fro() == pytest.approx(expect)
    again = m.predict(src)
    assert all(np.allclose(again.modules[n].dense(), pred.modules[n].dense()) for n in pred.modules)


def test_map_save_load_roundtrip(tmp_path):
    w = synthetic_world()
    m = _fit("svd_linear", w)
    m.save(tmp_path / "map")
    m2 = PairMap.load(tmp_path / "map")
    p1, p2 = m.predict(w["src"]["T10"]), m2.predict(w["src"]["T10"])
    for n in p1.modules:
        assert np.allclose(p1.modules[n].dense(), p2.modules[n].dense(), atol=1e-4)
    assert m2.split.holdout == ("T9", "T10")
