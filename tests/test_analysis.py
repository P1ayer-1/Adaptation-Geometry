"""RecoveredLift + predeclared gate logic on synthetic evaluation results (spec §5.5-5.6)."""

import json
import zlib

import numpy as np
import pytest

from uag.analysis import analyze
from uag.config import (BaseConfig, ExperimentConfig, GateSettings, HoldoutSplitSpec, LoraSettings, MapSettings,
                        TrainSettings)
from uag.pipeline import Paths, declare_gate

TASKS = ["T1_sentiment", "T3_paraphrase", "T8_arithmetic"]


def make_exp(tmp_path, all_tasks, **gate_kw):
    bases = [BaseConfig(name=n, model_id=n, source="local", family=f)
             for n, f in [("A", "fam1"), ("B", "fam2"), ("C", "fam3")]]
    tasks = [t for t in all_tasks if t.task_id in TASKS]
    return ExperimentConfig(
        name="synthetic", bases=bases, tasks=tasks, lora=LoraSettings(), train=TrainSettings(), seeds=[0, 1],
        splits=[HoldoutSplitSpec("s1", ["T8_arithmetic"]), HoldoutSplitSpec("s2", ["T1_sentiment"])],
        maps=MapSettings(), gate=GateSettings(bootstrap_samples=300, **gate_kw), require_pinned_revisions=False,
        artifacts_dir=str(tmp_path / "artifacts"), results_dir=str(tmp_path / "results"))


def write(paths, eval_id, role, target, task, primary, n=50, seed_=0, **meta):
    rng = np.random.default_rng(zlib.crc32(eval_id.encode()))
    scores = (rng.random(n) < primary).astype(float)
    paths.raw.mkdir(parents=True, exist_ok=True)
    (paths.raw / f"{eval_id}.jsonl").write_text("\n".join(json.dumps({"example_id": i, "score": s})
                                                          for i, s in enumerate(scores)))
    summ = {"eval_id": eval_id, "role": role, "target_base": target, "task_id": task, "eval_task": task,
            "template": "v1", "primary": float(scores.mean()), "n": n, **meta}
    (paths.raw / f"{eval_id}.summary.json").write_text(json.dumps(summ))


def populate(exp, learned_score, baseline_score):
    paths = Paths(exp)
    splits = {"s1": ["T8_arithmetic"], "s2": ["T1_sentiment"]}
    for b in ["A", "B", "C"]:
        for t in TASKS:
            write(paths, f"base_{b}_{t}", "base", b, t, 0.3, n=400)
            for seed in exp.seeds:
                write(paths, f"direct_{b}_{t}_{seed}", "direct", b, t, 0.9, n=400, seed=seed)
    for s in ["A", "B", "C"]:
        for tgt in ["A", "B", "C"]:
            if s == tgt:
                continue
            for split, hold in splits.items():
                for task in hold:
                    for seed in exp.seeds:
                        for m, sc in [("svd_procrustes", learned_score), ("svd_linear", learned_score),
                                      ("identity", baseline_score), ("cross_lora", baseline_score),
                                      ("random", 0.3)]:
                            write(paths, f"tr_{s}_{tgt}_{split}_{task}_{m}_{seed}", "transfer", tgt, task, sc,
                                  n=400, seed=seed, method=m, source_base=s, split_id=split)


def test_gate_passes_when_learned_map_recovers_lift(tmp_path, all_tasks):
    exp = make_exp(tmp_path, all_tasks)
    declare_gate(exp)
    populate(exp, learned_score=0.75, baseline_score=0.35)
    res = analyze(exp)
    g = res["gate"]
    assert g["passed"], g["criteria"]
    assert g["criteria"]["median_recovered_lift"]["value"] == pytest.approx(0.75, abs=0.1)
    memo = (tmp_path / "results" / "synthetic_decision.md").read_text()
    assert "**GO**" in memo and "Headroom" in memo
    assert (tmp_path / "results" / "tables" / "synthetic_cells.csv").exists()


def test_gate_fails_when_learned_map_equals_baseline(tmp_path, all_tasks):
    exp = make_exp(tmp_path, all_tasks)
    populate(exp, learned_score=0.45, baseline_score=0.45)
    g = analyze(exp)["gate"]
    assert not g["passed"]
    assert not g["criteria"]["median_recovered_lift"]["passed"]
    assert not g["criteria"]["beats_strongest_baseline"]["passed"]


def test_gate_cannot_be_edited_after_declaration(tmp_path, all_tasks):
    exp = make_exp(tmp_path, all_tasks)
    declare_gate(exp)
    exp2 = make_exp(tmp_path, all_tasks, min_median_recovered_lift=0.1)
    with pytest.raises(RuntimeError, match="changed after declaration"):
        declare_gate(exp2)
