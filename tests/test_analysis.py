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


def populate(exp, learned_score, baseline_score, zeroshot=0.1, fewshot=0.3):
    paths = Paths(exp)
    splits = {"s1": ["T8_arithmetic"], "s2": ["T1_sentiment"]}
    for b in ["A", "B", "C"]:
        for t in TASKS:
            write(paths, f"base_{b}_{t}", "base", b, t, zeroshot, n=400)
            write(paths, f"basefs_{b}_{t}", "base_fewshot", b, t, fewshot, n=400, shots=exp.fewshot.k,
                  fewshot_seed=exp.fewshot.seed)
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


def test_fewshot_baseline_sets_eligibility_and_flags(tmp_path, all_tasks):
    """Lift is measured against the few-shot base; a format-only adapter is ineligible and flagged."""
    from uag.analysis import headroom_rows, load_summaries

    exp = make_exp(tmp_path, all_tasks)
    # Zero-shot 0.1, few-shot 0.88, direct 0.9: 0.8 lift over zero-shot, but nearly all of it
    # comes from showing examples (format), and the few-shot base is near ceiling.
    populate(exp, learned_score=0.89, baseline_score=0.88, zeroshot=0.1, fewshot=0.88)
    res = analyze(exp)
    rows = headroom_rows(exp, load_summaries(Paths(exp)))
    assert rows and all(r["format_dominated"] and not r["eligible"] for r in rows)
    assert res["gate"] is None or not res["gate"]["passed"]
    memo = (tmp_path / "results" / "synthetic_decision.md").read_text()
    assert "format" in memo and "Few-shot" in memo

    # The same data judged against the zero-shot base would look like a large, eligible lift.
    exp_zs = make_exp(tmp_path / "zs", all_tasks, baseline="zeroshot")
    populate(exp_zs, learned_score=0.89, baseline_score=0.88, zeroshot=0.1, fewshot=0.88)
    assert all(r["eligible"] for r in headroom_rows(exp_zs, load_summaries(Paths(exp_zs))))


def test_fewshot_shots_are_deterministic_from_train(all_tasks, tmp_path):
    import dataclasses

    from uag.data import generate_dataset
    from uag.evaluate import select_shots
    from uag.prompts import build_fewshot_prompt, build_prompt

    t = next(x for x in all_tasks if x.task_id == "T2_nli")
    t = dataclasses.replace(t, n_train=200, n_valid=20, n_test=20)
    generate_dataset(t, tmp_path)
    a, b = select_shots(t, 5, 0, tmp_path), select_shots(t, 5, 0, tmp_path)
    assert [x["example_id"] for x in a] == [x["example_id"] for x in b]
    assert all(x["split"] == "train" for x in a)
    assert {x["target"] for x in a} == {"entailment", "neutral", "contradiction"}  # stratified
    assert [x["example_id"] for x in select_shots(t, 5, 1, tmp_path)] != [x["example_id"] for x in a]
    assert build_fewshot_prompt("I", [], "q") == build_prompt("I", "q")
    p = build_fewshot_prompt("I", a, "q")
    assert p.count("Input:") == 6 and p.endswith("Input: q\nOutput:")


def test_graded_criterion_requires_beating_target_mean(tmp_path, all_tasks):
    """Criterion 4: the learned map must beat the strongest baseline on RecoveredNLL."""
    exp = make_exp(tmp_path, all_tasks, graded_beats_baselines=True,
                   non_learned_baselines=["identity", "cross_lora", "random", "target_mean"])
    declare_gate(exp)
    populate(exp, learned_score=0.75, baseline_score=0.35)
    paths = Paths(exp)

    def graded(learned, mean):
        rows = []
        for s in ["A", "B", "C"]:
            for t in ["A", "B", "C"]:
                if s == t:
                    continue
                for task in ["T8_arithmetic", "T1_sentiment"]:
                    for seed in exp.seeds:
                        for m, v in [("svd_procrustes", learned), ("target_mean", mean), ("random", 0.0)]:
                            rows.append({"source": s, "target": t, "task": task, "seed": seed, "split": "x",
                                         "method": m, "recovered_nll": v + 0.01 * seed})
        (paths.exp_results / "graded_transfer.json").write_text(json.dumps({"rows": rows}))

    graded(0.4, 0.2)
    g = analyze(exp)["gate"]
    assert g["criteria"]["graded_beats_baselines"]["passed"]
    assert g["criteria"]["graded_beats_baselines"]["strongest_baseline"] == "target_mean" and g["passed"]
    graded(0.1, 0.2)
    g = analyze(exp)["gate"]
    assert not g["criteria"]["graded_beats_baselines"]["passed"] and not g["passed"]
    (paths.exp_results / "graded_transfer.json").unlink()
    assert not analyze(exp)["gate"]["criteria"]["graded_beats_baselines"]["passed"]


def test_late_gate_field_keeps_old_declarations(tmp_path, all_tasks):
    exp = make_exp(tmp_path, all_tasks)
    d1 = declare_gate(exp)
    assert "graded_beats_baselines" not in d1["gate"]  # omitted at its default: old hashes unchanged
    exp2 = make_exp(tmp_path, all_tasks, graded_beats_baselines=True)
    with pytest.raises(RuntimeError, match="changed after declaration"):
        declare_gate(exp2)
