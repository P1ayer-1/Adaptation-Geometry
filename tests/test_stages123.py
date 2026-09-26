"""Stage 1-3 building blocks on synthetic worlds with a known shared latent geometry."""

import numpy as np
import pytest

from uag.composition import CompositionOutcome, delta_combine, lerp, norm_matched, slerp
from uag.config import load_task
from uag.data import generate_dataset, load_split
from uag.extract_delta import DeltaSet, LowRank
from uag.geometry import (identity_test, layer_localization, rotate_update, seed_invariance, shuffled_pairing)
from uag.latent_model import LatentModel, leave_one_model_out, leave_one_task_out
from uag.metrics import get_metric
from uag.spectral import relative_error

K = 5
MODELS = {"m1": (20, 18, 2), "m2": (26, 14, 2), "m3": (22, 22, 3)}  # (d_out, d_in, layers)


def _orth(rng, d, k):
    return np.linalg.qr(rng.normal(size=(d, k)))[0]


def world(latents: dict[str, np.ndarray], seed=0):
    """Every model's module (l, q) update is A_{m,l} Z_t B_{m,l}ᵀ."""
    rng = np.random.default_rng(seed)
    out = {}
    for m, (do, di, L) in MODELS.items():
        AB = {l: (_orth(rng, do, K), _orth(rng, di, K), rng.uniform(0.5, 2)) for l in range(L)}
        out[m] = {}
        for t, z in latents.items():
            mods, info = {}, {}
            for l, (A, B, s) in AB.items():
                n = f"model.layers.{l}.self_attn.q_proj"
                mods[n] = LowRank.from_dense(s * A @ z @ B.T)
                info[n] = {"layer": l, "cls": "q"}
            out[m][t] = DeltaSet({"task_id": t, "num_layers": L, "model_type": "llama"}, mods, info)
    return out


def rel_err(pred: DeltaSet, true: DeltaSet) -> float:
    return float(np.mean([relative_error(pred.modules[n], true.modules[n]) for n in true.modules]))


def test_loto_bilinear_generalises_to_unseen_task():
    rng = np.random.default_rng(1)
    Z = {f"T{i}": rng.normal(size=(K, K)) for i in range(10)}
    w = world(Z)
    res = leave_one_task_out(w, "T9", ["m1"], decoder="bilinear", d_z=K * K, k=K, ridge=1e-6)
    for m, pred in res["predictions"].items():
        assert rel_err(pred, w[m]["T9"]) < 0.05, m


def test_loto_linear_needs_latent_in_span():
    rng = np.random.default_rng(2)
    E = [rng.normal(size=(K, K)) for _ in range(3)]
    Z = {f"T{i}": sum(rng.normal() * e for e in E) for i in range(10)}  # 3-dim task manifold
    w = world(Z)
    res = leave_one_task_out(w, "T9", ["m1", "m2"], decoder="linear", d_z=3, k=K, ridge=1e-8)
    assert rel_err(res["predictions"]["m3"], w["m3"]["T9"]) < 0.05
    assert res["model"].train_reconstruction_error({m: {t: d for t, d in w[m].items() if t != "T9"}
                                                    for m in w}) < 1e-3


def test_lomo_fits_only_new_decoder():
    rng = np.random.default_rng(3)
    Z = {f"T{i}": rng.normal(size=(K, K)) for i in range(10)}
    w = world(Z)
    res = leave_one_model_out(w, "m3", fit_tasks=[f"T{i}" for i in range(6)], decoder="bilinear",
                              d_z=K * K, k=K, ridge=1e-6)
    errs = [rel_err(res["predictions"][t], w["m3"][t]) for t in res["test_tasks"]]
    assert res["test_tasks"] == ["T6", "T7", "T8", "T9"]
    assert max(errs) < 0.05


def test_composition_in_latent_space():
    rng = np.random.default_rng(4)
    Z = {f"T{i}": rng.normal(size=(K, K)) for i in range(10)}
    Z["AB"] = Z["T0"] + Z["T1"]
    w = world(Z)
    train = {m: {t: d for t, d in per.items() if t != "AB"} for m, per in w.items()}
    lm = LatentModel("bilinear", K * K, K, 1e-6, sorted(train["m1"])).fit(train)
    z_ab = lm.latents["T0"] + lm.latents["T1"]
    pred = lm.predict("m2", z_ab, {"task_id": "AB"})
    assert rel_err(pred, w["m2"]["AB"]) < 0.05
    lin = delta_combine([w["m2"]["T0"], w["m2"]["T1"]], [1.0, 1.0])
    assert rel_err(lin, w["m2"]["AB"]) < 1e-6  # linear ground truth => linear baseline exact too
    nm = norm_matched(lin, w["m2"]["T0"])
    assert nm.fro() == pytest.approx(w["m2"]["T0"].fro())


def test_interpolation_and_outcomes():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    assert np.allclose(slerp(a, b, 0), a) and np.allclose(slerp(a, b, 1), b)
    assert np.linalg.norm(slerp(a, b, 0.5)) == pytest.approx(1.5)
    assert np.allclose(lerp(a, b, 0.5), [0.5, 1.0])
    ok = CompositionOutcome({"arith": 0.8, "concise": 0.9}, {"arith": 0.85, "concise": 0.95},
                            {"arith": 0.5, "concise": 0.5})
    assert ok.success and not ok.traded_off
    bad = CompositionOutcome({"arith": 0.84, "concise": 0.1}, {"arith": 0.85, "concise": 0.95},
                             {"arith": 0.5, "concise": 0.5})
    assert not bad.success and bad.traded_off


def test_geometry_controls():
    rng = np.random.default_rng(5)
    Z = {f"T{i}": rng.normal(size=(K, K)) for i in range(4)}
    w = world(Z)["m1"]
    same = seed_invariance([w["T0"], w["T0"]])
    assert same["global_cosine"]["mean"] == pytest.approx(1.0)
    rot = seed_invariance([w["T0"], rotate_update(w["T0"])])
    assert rot["global_cosine"]["mean"] < 0.5
    # Dataset invariance: two "datasets" per task = small perturbations of the same update.
    d = {}
    for t in ("T0", "T1"):
        for ds_id in ("a", "b"):
            noise = {n: lr + LowRank.from_dense(0.05 * rng.normal(size=lr.shape)) for n, lr in w[t].modules.items()}
            d[(t, ds_id)] = DeltaSet({"task_id": t}, noise, w[t].info)
    assert identity_test(d)["transformation_dominates"]
    loc = layer_localization(w["T0"])
    assert sum(loc["by_depth"].values()) == pytest.approx(1.0) and loc["by_group"] == {"attention": 1.0}
    perm = shuffled_pairing({t: w[t] for t in ("T0", "T1", "T2")})
    assert all(perm[t].meta["task_id"] != t for t in perm)


def test_composition_task_gold(tmp_path):
    t = load_task("configs/tasks/composition/C1_arith_concise.yaml")
    generate_dataset(t)
    metric = get_metric(t.metric)
    for ex in load_split(t, "test")[:100]:
        assert metric(ex["target"], ex) == 1.0
        assert all(get_metric(m)(ex["target"], ex) == 1.0 for m in t.factor_metrics.values())
