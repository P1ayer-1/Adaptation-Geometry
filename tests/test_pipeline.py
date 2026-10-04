"""Acceptance tests from the spec's first work order (§20 steps 2-5)."""

import json

import numpy as np
import pytest
import torch
import yaml

from uag.alignment import discover_modules, match_modules
from uag.config import LoraSettings
from uag.data import instruction_for, load_split
from uag.evaluate import evaluate_examples, run_evaluation
from uag.extract_delta import DeltaSet, LowRank, applied_delta, extract_deltas, from_factors, \
    verify_delta_reconstruction
from uag.models import load_base_model, state_fingerprint
from uag.spectral import (effective_rank, principal_angles, save_spectral, subspace_overlap,
                          summarize_deltaset)
from uag.train_lora import build_peft_model, collate, encode_example, load_trained


def _batch(tok, task, n=4):
    rows = load_split(task, "valid")[:n]
    b = collate([encode_example(tok, instruction_for(task), e, "v1", 128) for e in rows], tok.pad_token_id)
    b.pop("labels")
    return b


def test_train_and_reload_into_frozen_base(trained_run):
    base, task, run_dir = trained_run
    m = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    for key in ("run_id", "base_model", "base_revision", "task_id", "dataset_version", "seed", "lora_rank",
                "lora_alpha", "lora_dropout", "target_modules", "dtype", "optimizer", "learning_rate",
                "max_tokens_seen", "code_commit", "hardware", "dataset_sha256"):
        assert key in m, key
    assert m["base_frozen_fingerprint"]["unchanged"]
    assert m["tokens_seen"] >= m["max_tokens_seen"]
    assert m["selection"]["split"] == "valid"
    # Reload into the exact frozen base: same logits as a second independent reload.
    pm1, tok = load_trained(base, run_dir, device="cpu")
    pm2, _ = load_trained(base, run_dir, device="cpu")
    b = _batch(tok, task)
    with torch.no_grad():
        assert torch.equal(pm1(**b).logits, pm2(**b).logits)
    assert (run_dir / "adapter" / "adapter_model.safetensors").exists()


def test_delta_reconstruction_matches_peft_forward(trained_run):
    base, task, run_dir = trained_run
    pm, tok = load_trained(base, run_dir, device="cpu")
    ds = extract_deltas(run_dir / "adapter", pm.config.model_type)
    rep = verify_delta_reconstruction(pm, ds, _batch(tok, task))
    assert rep["passed"], rep
    assert rep["adapter_effect_max_abs"] > 10 * rep["max_abs_logit_diff"]
    assert all(lr.rank <= 4 for lr in ds.modules.values())


@pytest.mark.parametrize("base_name,rslora", [("tiny_qwen2_b", True), ("tiny_phi_c", False)])
def test_delta_reconstruction_random_lora(tiny_bases, all_tasks, tmp_path, base_name, rslora):
    """Non-zero random B so the check is not trivial; covers rsLoRA scaling and Phi naming."""
    model, tok, _ = load_base_model(tiny_bases[base_name], dtype="float32", device="cpu")
    pm, inv = build_peft_model(model, LoraSettings(rank=4, alpha=16, use_rslora=rslora))
    torch.manual_seed(0)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    pm.save_pretrained(tmp_path / "adapter")
    ds = extract_deltas(tmp_path / "adapter", model.config.model_type)
    assert set(ds.modules) == set(inv.names())
    rep = verify_delta_reconstruction(pm, ds, _batch(tok, all_tasks[1]))
    assert rep["passed"], rep
    if base_name == "tiny_phi_c":
        assert "gate" in inv.omissions


def test_applied_delta_restores_weights(tiny_bases):
    model, _, _ = load_base_model(tiny_bases["tiny_llama_a1"], dtype="float32", device="cpu")
    inv = discover_modules(model)
    rng = np.random.default_rng(0)
    mods = {m.name: from_factors(rng.normal(size=(m.out_features, 2)), rng.normal(size=(2, m.in_features)))
            for m in inv.modules}
    ds = DeltaSet({"task_id": "x"}, mods)
    before = state_fingerprint(model)
    with applied_delta(model, ds):
        assert state_fingerprint(model) != before
    assert state_fingerprint(model) == before


def test_evaluate_base_and_adapter_from_manifests(trained_run, tmp_path):
    base, task, run_dir = trained_run
    s_base = run_evaluation(base, task, tmp_path, "base", split="test", max_examples=16, device="cpu")
    s_ad = run_evaluation(base, task, tmp_path, "adapter", adapter_dir=run_dir / "adapter", split="test",
                          max_examples=16, device="cpu")
    rows = [json.loads(l) for l in (tmp_path / "adapter.jsonl").read_text().splitlines()]
    assert len(rows) == 16 and {"example_id", "prediction", "score"} <= set(rows[0])
    assert s_base["kind"] == "base" and s_ad["kind"] == "adapter"
    assert 0.0 <= s_ad["primary"] <= 1.0 and "macro_f1" in s_ad
    # Evaluating a delta equal to the adapter's reconstruction gives identical per-example output.
    ds = extract_deltas(run_dir / "adapter", "llama")
    ds.save(tmp_path / "delta")
    run_evaluation(base, task, tmp_path, "delta", delta_dir=tmp_path / "delta", split="test",
                   max_examples=16, device="cpu")
    rows_d = [json.loads(l) for l in (tmp_path / "delta.jsonl").read_text().splitlines()]
    assert [r["prediction"] for r in rows] == [r["prediction"] for r in rows_d]


def test_generation_path(trained_run, all_tasks):
    base, _, run_dir = trained_run
    model, tok, _ = load_base_model(base, dtype="float32", device="cpu")
    task = next(t for t in all_tasks if t.task_id == "T10_format")
    recs, summ = evaluate_examples(model, tok, task, load_split(task, "test")[:3])
    assert len(recs) == 3 and all(isinstance(r["prediction"], str) for r in recs)


def test_spectral_tools(trained_run, tmp_path):
    _, _, run_dir = trained_run
    ds = extract_deltas(run_dir / "adapter", "llama")
    summ = save_spectral(ds, tmp_path / "spec")
    assert abs(sum(summ["energy_by_layer"].values()) - 1) < 1e-9
    assert DeltaSet.load(tmp_path / "spec").modules.keys() == ds.modules.keys()
    rng = np.random.default_rng(1)
    q = np.linalg.qr(rng.normal(size=(20, 3)))[0]
    assert np.allclose(principal_angles(q, q @ np.linalg.qr(rng.normal(size=(3, 3)))[0]), 0, atol=1e-6)
    assert subspace_overlap(q, np.linalg.qr(rng.normal(size=(20, 3)))[0]) < 0.9
    assert effective_rank(np.ones(5)) == pytest.approx(5)
    m = rng.normal(size=(12, 3)) @ rng.normal(size=(3, 9))
    lr = LowRank.from_dense(m)
    assert lr.rank == 3 and np.allclose(lr.dense(), m)
    f = from_factors(rng.normal(size=(12, 3)), rng.normal(size=(3, 9)), 2.0)
    assert np.allclose((f + f).dense(), 2 * f.dense())


def test_module_matching(tiny_bases):
    invs = {}
    for n in ("tiny_llama_a1", "tiny_llama_a2", "tiny_phi_c"):
        model, _, _ = load_base_model(tiny_bases[n], dtype="float32", device="cpu")
        invs[n] = discover_modules(model)
    same = match_modules(invs["tiny_llama_a1"], invs["tiny_llama_a1"])
    assert same.bijective
    deeper = match_modules(invs["tiny_llama_a1"], invs["tiny_llama_a2"])
    assert not deeper.bijective and len(deeper.pairs) == len(invs["tiny_llama_a2"].modules)
    phi = match_modules(invs["tiny_llama_a1"], invs["tiny_phi_c"])
    assert any(o["class"] == "gate" for o in phi.omissions)
    assert all(s.cls == t.cls for s, t in phi.pairs)


def test_gradient_checkpointing_training(tiny_bases, all_tasks, tiny_train_settings, tmp_path):
    """The 8 GB dry-run settings (gradient checkpointing) must still train and verify."""
    import dataclasses

    from uag.train_lora import train_lora

    task = next(t for t in all_tasks if t.task_id == "T4_json")
    train = dataclasses.replace(tiny_train_settings, gradient_checkpointing=True, batch_size=2, grad_accum=2)
    run_dir = train_lora(tiny_bases["tiny_qwen2_b"], task, LoraSettings(rank=4, alpha=8), train, 0, tmp_path)
    m = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    assert not m["diverged"] and m["selection"]["value"] < m["selection"]["initial"]["valid_loss"]
    pm, tok = load_trained(tiny_bases["tiny_qwen2_b"], run_dir, device="cpu")
    ds = extract_deltas(run_dir / "adapter", "qwen2")
    assert verify_delta_reconstruction(pm, ds, _batch(tok, task))["passed"]


def test_factor_movement_recorded(trained_run):
    _, _, run_dir = trained_run
    m = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    fm = m["factor_movement"]["overall"]
    assert fm["a_rel_move_mean"] > 0 and fm["b_norm_mean"] > 0
    assert 0 <= fm["a_rowspace_overlap_mean"] <= 1 + 1e-9
    assert set(m["factor_movement"]["by_class"]) <= {"q", "k", "v", "o", "up", "down", "gate"}
    assert (run_dir / "factor_movement.json").exists()
    log = [json.loads(l) for l in (run_dir / "train_log.jsonl").read_text().splitlines()]
    assert any("a_rel_move_mean" in r for r in log)


@pytest.mark.parametrize("scope", ["seed", "task"])
def test_seed_comparison_detects_shared_init(tmp_path, tiny_bases, all_tasks, tiny_train_settings, scope):
    """With a shared seed, different tasks share A0 and hence ΔW's input directions; with
    init_seed_scope: task they do not."""
    import dataclasses

    from uag.config import ExperimentConfig
    from uag.diagnostics import render_geometry_checks, seed_comparison
    from uag.pipeline import train_grid

    tasks = [t for t in all_tasks if t.task_id in ("T2_nli", "T3_paraphrase")]
    exp = ExperimentConfig(name="seedcmp", bases=[tiny_bases["tiny_llama_a1"]], tasks=tasks,
                           lora=LoraSettings(rank=4, alpha=8, init_seed_scope=scope),
                           train=dataclasses.replace(tiny_train_settings, max_tokens_seen=1500),
                           seeds=[0, 1], require_pinned_revisions=False,
                           artifacts_dir=str(tmp_path / "a"), results_dir=str(tmp_path / "r"))
    train_grid(exp)
    r = seed_comparison(exp)["bases"]["tiny_llama_a1"]
    assert r["same_task_diff_seed"]["n_pairs"] == 2 and r["diff_task_same_seed"]["n_pairs"] == 2
    shared = r["diff_task_same_seed"]["right_overlap"]
    ref = r["diff_task_diff_seed"]["right_overlap"]
    if scope == "seed":
        assert shared > 0.9 and shared > 2 * ref
    else:
        assert shared < 0.5
    text = "\n".join(render_geometry_checks(exp))
    assert "Seed comparison" in text and "Factor movement" in text


def test_finished_runs_are_not_reused_under_changed_settings(trained_run, tiny_train_settings):
    import dataclasses

    from uag.train_lora import StaleRunError, train_lora

    base, task, run_dir = trained_run
    lora = LoraSettings(rank=4, alpha=8)
    assert train_lora(base, task, lora, tiny_train_settings, 0, run_dir.parent) == run_dir  # reuse is fine
    with pytest.raises(StaleRunError, match="init_seed_scope"):
        train_lora(base, task, dataclasses.replace(lora, init_seed_scope="task"), tiny_train_settings, 0,
                   run_dir.parent)
    with pytest.raises(StaleRunError, match="learning_rate"):
        train_lora(base, task, lora, dataclasses.replace(tiny_train_settings, learning_rate=1e-5), 0, run_dir.parent)


def test_early_stopping_records_stop_and_budget_proposal(tmp_path, tiny_bases, all_tasks, tiny_train_settings):
    import dataclasses

    from uag.budget import curve_summary, learning_curves, render_learning_curves
    from uag.config import ExperimentConfig
    from uag.train_lora import train_lora

    task = next(t for t in all_tasks if t.task_id == "T2_nli")
    # min_delta so large that no evaluation counts as an improvement: the first eval (step 5) sets the
    # reference, steps 10 and 15 are the two evals without improvement -> stop at 15.
    train = dataclasses.replace(tiny_train_settings, max_tokens_seen=200_000, early_stopping_patience=2,
                                early_stopping_min_delta=10.0, min_steps=10, eval_every_steps=5)
    run_dir = train_lora(tiny_bases["tiny_llama_a1"], task, LoraSettings(rank=4, alpha=8), train, 0, tmp_path / "runs")
    m = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    st = m["stopping"]
    assert st["reason"] == "early_stopping" and st["step"] == 15 and m["tokens_seen"] < 200_000
    assert st["best_step"] <= st["step"] and m["selection"]["step"] == st["best_step"]

    s = curve_summary([(0, 0, 1.0), (5, 100, 0.5), (10, 200, 0.2), (15, 300, 0.19)], lower_is_better=True)
    assert s["t50"] == 100 and s["t95"] == 200 and s["best_tokens"] == 300 and s["cap_binding"]
    exp = ExperimentConfig(name="curves", bases=[tiny_bases["tiny_llama_a1"]], tasks=[task],
                           lora=LoraSettings(rank=4, alpha=8), train=train, seeds=[0], require_pinned_revisions=False,
                           artifacts_dir=str(tmp_path / "a"), results_dir=str(tmp_path / "r"))
    import shutil

    from uag.pipeline import Paths

    shutil.copytree(run_dir, Paths(exp).run_dir("tiny_llama_a1", "T2_nli", 0))
    res = learning_curves(exp)
    assert res["runs"][0]["stop_reason"] == "early_stopping"
    assert "Proposed per-task token cap" in "\n".join(render_learning_curves(res))


def test_gpu_benchmark_runs_and_prices(tmp_path):
    from uag.benchmark import render_benchmark, run_benchmark

    r = run_benchmark("configs/bases/tiny_llama_a1.yaml", "configs/tasks/v2/T8_arithmetic.yaml",
                      "configs/train/smoke.yaml", "configs/lora/r4.yaml", minutes=1e-4, price_per_hour=2.0,
                      eval_examples=4, out_root=str(tmp_path))
    assert r["stop_reason"] == "wall_clock" and r["tokens_per_second"] > 0
    assert r["cost_per_run"] == pytest.approx(r["hours_per_run"] * 2.0)
    assert "Stage-0 grid" in render_benchmark(r)


def test_no_silent_cpu_fallback_when_gpus_present(monkeypatch):
    import shutil

    from uag.models import pick_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.delenv("UAG_ALLOW_CPU", raising=False)
    with pytest.raises(RuntimeError, match="cannot use CUDA"):
        pick_device("auto")
    monkeypatch.setenv("UAG_ALLOW_CPU", "1")
    assert pick_device("auto").type == "cpu"
    assert pick_device("cpu").type == "cpu"


def test_frozen_shared_A_per_base(tmp_path, tiny_bases, all_tasks, tiny_train_settings):
    """train_A: false + init_seed_scope: base -> every task and seed of a base shares one fixed A."""
    import dataclasses

    from safetensors.torch import load_file

    from uag.config import ConfigError
    from uag.train_lora import train_lora

    with pytest.raises(ConfigError):
        LoraSettings(train_A=False, init_seed_scope="task").validate()
    lora = LoraSettings(rank=4, alpha=8, init_seed_scope="base", train_A=False)
    lora.validate()
    train = dataclasses.replace(tiny_train_settings, max_tokens_seen=1500)
    tasks = [t for t in all_tasks if t.task_id in ("T2_nli", "T3_paraphrase")]
    runs = [train_lora(tiny_bases["tiny_llama_a1"], t, lora, train, s, tmp_path) for t in tasks for s in (0, 1)]
    weights = [load_file(str(r / "adapter" / "adapter_model.safetensors")) for r in runs]
    a_keys = [k for k in weights[0] if ".lora_A." in k]
    for w in weights[1:]:
        assert all(torch.equal(w[k], weights[0][k]) for k in a_keys)  # same A everywhere
    b_keys = [k for k in weights[0] if ".lora_B." in k]
    assert any(not torch.equal(weights[0][k], weights[2][k]) for k in b_keys)  # B is task-specific
    m = yaml.safe_load((runs[0] / "manifest.yaml").read_text())
    assert m["lora_train_A"] is False and m["factor_movement"]["overall"]["a_rel_move_max"] == 0.0
    other = train_lora(tiny_bases["tiny_qwen2_b"], tasks[0], lora, train, 0, tmp_path)
    assert yaml.safe_load((other / "manifest.yaml").read_text())["lora_init_seed"] != m["lora_init_seed"]


def test_min_epochs_delays_early_stopping(tmp_path, tiny_bases, all_tasks, tiny_train_settings):
    import dataclasses

    from uag.train_lora import train_lora

    task = next(t for t in all_tasks if t.task_id == "T2_nli")
    task = dataclasses.replace(task, n_train=120, n_valid=20, n_test=20)
    from uag.data import generate_dataset

    generate_dataset(task, tmp_path / "data")
    train = dataclasses.replace(tiny_train_settings, max_tokens_seen=10**6, early_stopping_patience=1,
                                early_stopping_min_delta=10.0, eval_every_steps=5, min_epochs=2.0)
    run = train_lora(tiny_bases["tiny_llama_a1"], task, LoraSettings(rank=4, alpha=8), train, 0, tmp_path / "r",
                     data_dir=tmp_path / "data")
    st = yaml.safe_load((run / "manifest.yaml").read_text())["stopping"]
    assert st["min_steps"] == 60 and st["reason"] == "early_stopping" and st["step"] >= 60  # 2 x 120 / 4


def test_cli_exits_fast_on_cuda_failure(monkeypatch):
    import os

    from uag import cli

    def boom(argv=None):
        raise RuntimeError("CUDA error: unknown error")

    codes = []
    monkeypatch.setattr(cli, "_main", boom)
    monkeypatch.setattr(os, "_exit", lambda c: codes.append(c))  # the stub returns, so the error re-raises
    with pytest.raises(RuntimeError):
        cli.main([])
    assert codes == [3]
    monkeypatch.setattr(cli, "_main", lambda argv=None: (_ for _ in ()).throw(ValueError("other")))
    with pytest.raises(ValueError):
        cli.main([])


def test_gold_nll_and_recovered_nll(trained_run):
    """Applying the direct adapter's own ΔW lowers the gold NLL exactly as the adapter does."""
    from uag.extract_delta import applied_delta
    from uag.graded import gold_nll

    base, task, run_dir = trained_run
    pm, tok = load_trained(base, run_dir, device="cpu", dtype="float32")
    rows = load_split(task, "test")[:8]
    nll_adapter = gold_nll(pm, tok, task, rows)
    with pm.disable_adapter():
        nll_base = gold_nll(pm, tok, task, rows)
        with applied_delta(pm, extract_deltas(run_dir / "adapter", "llama")):
            nll_delta = gold_nll(pm, tok, task, rows)
    assert nll_delta == pytest.approx(nll_adapter, abs=1e-3)
    assert nll_adapter < nll_base


def test_eye_matrix_kinds():
    from uag.eyes import eye_matrix

    g = torch.Generator().manual_seed(0)
    d, r = 12, 3
    q = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    evals = torch.tensor([100.0] * 2 + [0.01] * (d - 2))
    sigma = q @ torch.diag(evals) @ q.T
    a0 = torch.randn(r, d, generator=g)
    energy = lambda a: float(torch.trace(a.double() @ sigma.double() @ a.double().T))  # noqa: E731
    pca = eye_matrix(sigma, a0, "pca", g)
    top = q[:, :2]
    assert torch.allclose((pca[:2] / pca[:2].norm(dim=1, keepdim=True)).abs() @ top.abs(), torch.eye(2), atol=1e-3) or \
        float(((pca[:2] @ top) ** 2).sum() / (pca[:2] ** 2).sum()) > 0.99
    wh = eye_matrix(sigma, a0, "whitened", g)
    assert energy(pca) == pytest.approx(energy(a0), rel=1e-4) and energy(wh) == pytest.approx(energy(a0), rel=1e-4)
    # whitening: the faint directions carry far more of A's weight than under random A
    faint = q[:, 2:]
    share = lambda a: float(((a @ faint) ** 2).sum() / (a ** 2).sum())  # noqa: E731
    assert share(wh) > 0.9


def test_smart_eyes_training(tmp_path, tiny_bases, all_tasks, tiny_train_settings):
    import dataclasses

    from safetensors.torch import load_file

    from uag.train_lora import train_lora

    lora = LoraSettings(rank=4, alpha=8, init_seed_scope="base", train_A=False, a_init="whitened",
                        a_calibration="builtin", a_calibration_tokens=2000)
    train = dataclasses.replace(tiny_train_settings, max_tokens_seen=1500)
    tasks = [t for t in all_tasks if t.task_id in ("T2_nli", "T3_paraphrase")]
    runs = [train_lora(tiny_bases["tiny_llama_a1"], t, lora, train, 0, tmp_path / "runs") for t in tasks]
    w = [load_file(str(r / "adapter" / "adapter_model.safetensors")) for r in runs]
    a_keys = [k for k in w[0] if ".lora_A." in k]
    assert all(torch.equal(w[0][k], w[1][k]) for k in a_keys)  # one set of eyes per base
    m = yaml.safe_load((runs[0] / "manifest.yaml").read_text())
    assert m["lora_a_init"] == "whitened" and m["lora_eyes"]["sha256"]
    assert len(list((tmp_path / "eyes").glob("*.safetensors"))) == 1  # computed once, then cached
    rnd = train_lora(tiny_bases["tiny_llama_a1"], tasks[0],
                     LoraSettings(rank=4, alpha=8, init_seed_scope="base", train_A=False), train, 0, tmp_path / "rnd")
    w_rnd = load_file(str(rnd / "adapter" / "adapter_model.safetensors"))
    assert not torch.equal(w_rnd[a_keys[0]], w[0][a_keys[0]])
    pm, tok = load_trained(tiny_bases["tiny_llama_a1"], runs[0], device="cpu")
    ds = extract_deltas(runs[0] / "adapter", "llama")
    assert verify_delta_reconstruction(pm, ds, _batch(tok, tasks[0]))["passed"]


def test_shared_adapter_end_to_end(tmp_path, tiny_bases, all_tasks, tiny_train_settings):
    import dataclasses

    from uag.config import ExperimentConfig, HoldoutSplitSpec
    from uag.extract_delta import applied_delta
    from uag.shared_adapter import (SharedSettings, SharedSystem, evaluate_transfer, render, train_connectors,
                                    train_core)
    from uag.transfer import HoldoutLeakError

    tasks = [dataclasses.replace(t, n_train=200, n_valid=8, n_test=8) for t in all_tasks
             if t.task_id in ("T2_nli", "T3_paraphrase", "T9_clinical")]
    from uag.data import generate_dataset

    for t in tasks:
        generate_dataset(t, tmp_path / "data")
    train = dataclasses.replace(tiny_train_settings, batch_size=2, valid_max_examples=4, max_tokens_seen=600,
                                eval_every_steps=2)
    exp = ExperimentConfig(name="shared", bases=[tiny_bases["tiny_llama_a1"], tiny_bases["tiny_qwen2_b"]],
                           tasks=tasks, lora=LoraSettings(), train=train, seeds=[0],
                           splits=[HoldoutSplitSpec("h", ["T9_clinical"])], require_pinned_revisions=False,
                           eval_max_examples=4, artifacts_dir=str(tmp_path / "a"), results_dir=str(tmp_path / "r"),
                           data_dir=str(tmp_path / "data"))
    s = SharedSettings(d_shared=4, connector_steps=4, eval_every=2, graded_examples=4)
    system = SharedSystem(exp, s, device="cpu")
    out = tmp_path / "shared"
    out.mkdir()
    with pytest.raises(HoldoutLeakError):
        train_connectors(system, ["T2_nli", "T9_clinical"], ["T9_clinical"], out)
    meta = train_connectors(system, ["T2_nli", "T3_paraphrase"], ["T9_clinical"], out)
    assert meta["best_step"] in (2, 4) and (out / "connectors.safetensors").exists()
    assert any(float(q.abs().sum()) > 0 for q in system.Q.values())  # connectors trained
    for b in system.models:
        train_core(system, "T9_clinical", b, out / "heldout" / f"T9_clinical__on_{b}.safetensors")
    # hooks == adding the exact ΔW = scale·Q C P to the weights
    cores = system.new_cores("random", like=system.new_cores("identity"), gen=torch.Generator().manual_seed(0))
    b = "tiny_llama_a1"
    batch = _batch(system.toks[b], tasks[0])
    with torch.no_grad():
        with system.active(b, cores) as model:
            hooked = model(**batch).logits
        with applied_delta(system.models[b], system.delta_set(b, cores)):
            merged = system.models[b](**batch).logits
    assert torch.allclose(hooked, merged, atol=1e-4)
    res = evaluate_transfer(system, ["T9_clinical"], out)
    conds = {r["condition"] for r in res["rows"]}
    assert "transferred from tiny_qwen2_b" in conds and "mean training core" in conds
    assert "RecNLL" in render(res)


def test_shared_adapter_zero_default_cores(tmp_path, tiny_bases, all_tasks, tiny_train_settings):
    """core_init: zero -> an untrained core is exactly the untouched base; training still works
    (also with gradient checkpointing)."""
    import dataclasses

    from uag.config import ExperimentConfig, HoldoutSplitSpec
    from uag.data import generate_dataset
    from uag.shared_adapter import SharedSettings, SharedSystem, train_connectors, train_core

    tasks = [dataclasses.replace(t, n_train=200, n_valid=8, n_test=8) for t in all_tasks
             if t.task_id in ("T2_nli", "T3_paraphrase", "T9_clinical")]
    for t in tasks:
        generate_dataset(t, tmp_path / "data")
    train = dataclasses.replace(tiny_train_settings, batch_size=2, valid_max_examples=4, max_tokens_seen=600,
                                eval_every_steps=2, gradient_checkpointing=True)
    exp = ExperimentConfig(name="shared0", bases=[tiny_bases["tiny_llama_a1"], tiny_bases["tiny_qwen2_b"]],
                           tasks=tasks, lora=LoraSettings(), train=train, seeds=[0],
                           splits=[HoldoutSplitSpec("h", ["T9_clinical"])], require_pinned_revisions=False,
                           eval_max_examples=4, artifacts_dir=str(tmp_path / "a"), results_dir=str(tmp_path / "r"),
                           data_dir=str(tmp_path / "data"))
    system = SharedSystem(exp, SharedSettings(d_shared=4, connector_steps=4, eval_every=2, core_init="zero"),
                          device="cpu")
    b = "tiny_llama_a1"
    batch = _batch(system.toks[b], tasks[0])
    with torch.no_grad():
        plain = system.models[b](**batch).logits
        with system.active(b, system.new_cores("zero")) as model:
            assert torch.equal(model(**batch).logits, plain)  # zero core = untouched base
    out = tmp_path / "shared"
    out.mkdir()
    train_connectors(system, ["T2_nli", "T3_paraphrase"], ["T9_clinical"], out)
    from safetensors.torch import load_file

    cores = load_file(str(out / "train_cores.safetensors"))
    assert any(float(v.abs().sum()) > 0 for v in cores.values())  # cores moved away from zero
    meta = train_core(system, "T9_clinical", b, out / "heldout" / "T9.safetensors")
    assert meta["steps"] >= 1


def test_shared_adapter_phase1_resumes_after_crash(tmp_path, tiny_bases, all_tasks, tiny_train_settings, monkeypatch,
                                                   capsys):
    import dataclasses

    import uag.shared_adapter as sa
    from uag.config import ExperimentConfig, HoldoutSplitSpec
    from uag.data import generate_dataset

    tasks = [dataclasses.replace(t, n_train=100, n_valid=4, n_test=4) for t in all_tasks
             if t.task_id in ("T2_nli", "T3_paraphrase", "T9_clinical")]
    for t in tasks:
        generate_dataset(t, tmp_path / "data")
    train = dataclasses.replace(tiny_train_settings, batch_size=2, valid_max_examples=4)
    exp = ExperimentConfig(name="resume", bases=[tiny_bases["tiny_llama_a1"], tiny_bases["tiny_qwen2_b"]],
                           tasks=tasks, lora=LoraSettings(), train=train, seeds=[0],
                           splits=[HoldoutSplitSpec("h", ["T9_clinical"])], require_pinned_revisions=False,
                           artifacts_dir=str(tmp_path / "a"), results_dir=str(tmp_path / "r"), data_dir=str(tmp_path / "data"))
    s = sa.SharedSettings(d_shared=4, connector_steps=6, eval_every=2, core_init="zero")
    out = tmp_path / "shared"
    out.mkdir()
    real_loss, calls = sa._loss, {"n": 0}

    def crashing(system, base, batch, train=False):
        if train:
            calls["n"] += 1
            if calls["n"] == 9:  # step 5 (2 bases per step): after the step-4 checkpoint
                raise RuntimeError("CUDA error: simulated")
        return real_loss(system, base, batch, train)

    monkeypatch.setattr(sa, "_loss", crashing)
    with pytest.raises(RuntimeError, match="simulated"):
        sa.train_connectors(sa.SharedSystem(exp, s, device="cpu"), ["T2_nli", "T3_paraphrase"], ["T9_clinical"], out)
    assert (out / "phase1_checkpoint.pt").exists()
    monkeypatch.setattr(sa, "_loss", real_loss)
    meta = sa.train_connectors(sa.SharedSystem(exp, s, device="cpu"), ["T2_nli", "T3_paraphrase"], ["T9_clinical"], out)
    assert "resuming from step 4" in capsys.readouterr().out
    assert meta["best_step"] in (2, 4, 6) and not (out / "phase1_checkpoint.pt").exists()
    steps = [json.loads(l)["step"] for l in (out / "connector_log.jsonl").read_text().splitlines()]
    assert steps[-1] == 6 and 5 in steps
