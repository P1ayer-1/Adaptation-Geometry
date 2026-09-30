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
