import dataclasses
import json

import numpy as np
import pytest

from uag.data import (DataIntegrityError, dataset_dir, generate_dataset, load_split, manifest_path,
                      read_jsonl, validate_example)
from uag.metrics import (bootstrap_ci, get_metric, json_field_f1, macro_f1, paired_bootstrap_diff,
                         recovered_lift)


def _shrink(task, n=40):
    return dataclasses.replace(task, n_train=n, n_valid=n // 2, n_test=n // 2)


def test_gold_targets_score_perfectly(panel_tasks, tmp_path):
    """Every task's own gold target must achieve metric 1.0 (metric/generator consistency)."""
    for task in panel_tasks:
        t = _shrink(task)
        generate_dataset(t, tmp_path)
        metric = get_metric(t.metric)
        for ex in load_split(t, "test", tmp_path):
            assert metric(ex["target"], ex) == 1.0, (t.task_id, ex)


def test_splits_disjoint_and_schema(panel_tasks, tmp_path):
    for task in panel_tasks:
        t = _shrink(task, 60)
        generate_dataset(t, tmp_path)
        inputs = {}
        for split in ("train", "valid", "test"):
            rows = load_split(t, split, tmp_path)
            for r in rows:
                validate_example(r)
            inputs[split] = {r["input"] for r in rows}
            assert len(inputs[split]) == len(rows)
        assert not inputs["train"] & inputs["test"]
        assert not inputs["valid"] & inputs["test"]
        assert not inputs["train"] & inputs["valid"]


def test_generation_is_deterministic_and_manifest_guards(small_task):
    t, d = small_task
    m1 = generate_dataset(t, d)
    m2 = generate_dataset(t, d)  # regenerating an existing version must match
    assert m1["train_sha256"] == m2["train_sha256"]
    # A generator change under the same version must be refused.
    changed = dataclasses.replace(t, seed=t.seed + 1)
    with pytest.raises(DataIntegrityError):
        generate_dataset(changed, d)


def test_tampering_detected(small_task):
    t, d = small_task
    generate_dataset(t, d)
    path = dataset_dir(t, d) / "test.jsonl"
    rows = read_jsonl(path)
    rows[0]["target"] = "tampered"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(DataIntegrityError):
        load_split(t, "test", d)
    assert manifest_path(t, d).exists()


def test_committed_manifests_match_generators(panel_tasks):
    """The committed manifests in data/manifests must be reproducible from the code."""
    for t in panel_tasks:
        for split in ("train", "valid", "test"):
            load_split(t, split)


def test_metric_edge_cases():
    ex = {"target": '{"name": "A B", "age": 3, "city": "X", "occupation": "y"}',
          "metadata": {"record": {"name": "A B", "age": 3, "city": "X", "occupation": "y"}}}
    assert json_field_f1('{"name": "A B", "age": 3}', ex) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert json_field_f1("not json", ex) == 0.0
    assert json_field_f1('{"bogus": 1}', ex) == 0.0
    fc = get_metric("format_constraint")
    ex10 = {"metadata": {"items": ["apple", "river"]}}
    assert fc("- APPLE\n- RIVER\nEND", ex10) == 1.0
    assert fc("- apple\n- RIVER\nEND", ex10) == 0.0
    cs = get_metric("concise_success")
    ex5 = {"metadata": {"answer": "Buenos Aires", "max_words": 4}}
    assert cs("Buenos Aires", ex5) == 1.0
    assert cs("The capital city is clearly Buenos Aires", ex5) == 0.0
    code = get_metric("unit_test_pass")
    ex7 = {"metadata": {"tests": ["assert f(2) == 4"]}}
    assert code("def f(x):\n    return 2 * x\nThat is it.", ex7) == 1.0
    assert code("def f(x):\n    while True: pass", {"metadata": {"tests": ["f(1)"]}}) == 0.0
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_recovered_lift_and_bootstrap():
    r = recovered_lift(0.6, 0.5, 0.9)
    assert r.eligible and r.recovered_lift == pytest.approx(0.25)
    assert not recovered_lift(0.6, 0.5, 0.52).eligible
    assert recovered_lift(5.0, 0.5, 0.9).capped
    pt, lo, hi = bootstrap_ci(np.ones(50))
    assert pt == lo == hi == 1.0
    d = paired_bootstrap_diff(np.ones(30), np.zeros(30))
    assert d["excludes_zero"] and d["diff"] == 1.0


def test_v1_generators_still_registered_and_unchanged(all_tasks, v2_tasks):
    """Redesigning a generator must not change v1: both versions stay registered side by side."""
    from uag.tasks import get_generator

    for t in all_tasks:
        assert get_generator(t.generator, t.generator_version).version == "1"
    for t in v2_tasks:
        assert t.version == 2 and get_generator(t.generator, t.generator_version).version == "2"
    with pytest.raises(DataIntegrityError):
        generate_dataset(dataclasses.replace(v2_tasks[0], generator_version="99"), "unused")


def test_v2_metrics_resist_shortcuts():
    from uag.tasks_v2 import clinical_term, format_v2_lines

    # T5/T6: listing every candidate does not count; the first committed answer does.
    ex = {"metadata": {"answer": "Chen", "candidates": ["Alice", "Bruno", "Chen"], "max_words": 4,
                       "min_words": 3, "max_words_v": 80}}
    cs = get_metric("concise_success_v2")
    assert cs("Chen", ex) == 1.0 and cs("Alice or Chen", ex) == 0.0
    exn = {"metadata": {"answer": "27", "candidates": None, "max_words": 4}}
    assert cs("27", exn) == 1.0 and cs("15 or 27", exn) == 0.0
    vs = get_metric("verbose_success_v2")
    exv = {"metadata": {"answer": "Chen", "candidates": ["Alice", "Chen"], "min_words": 5, "max_words": 80}}
    assert vs("Well, the answer is Chen because Alice has fewer.", exv) == 1.0
    assert vs("Well, Alice has fewer, so the answer is Chen.", exv) == 0.0
    # T4: nested fields are scored one leaf at a time; unknown top-level keys are a schema error.
    rec = {"name": "A B", "employer": {"name": "X", "city": "Y"}, "manager": None}
    exj = {"metadata": {"record": rec}}
    jf = get_metric("json_nested_f1")
    assert jf(json.dumps(rec), exj) == 1.0
    assert jf('{"name": "A B", "employer": {"name": "X", "city": "Z"}, "manager": null}', exj) == pytest.approx(0.75)
    assert jf('{"name": "A B", "extra": 1}', exj) == 0.0
    # T9 / T10 rules.
    assert clinical_term("gastr", "itis") == "gastritis" and clinical_term("cardi", "itis") == "carditis"
    assert clinical_term("arthr", "scopy") == "arthroscopy" and clinical_term("nephr", "ectomy") == "nephrectomy"
    assert format_v2_lines(["tiger", "apple", "quartz"]) == ["1. apple", "2. QUARTZ", "3. tiger", "TOTAL: 3"]


def test_code_with_template_leading_space_runs():
    """Answers after 'Output:' start with a space; that must not make valid code fail."""
    code = get_metric("unit_test_pass")
    assert code(" def f(x):\n    return 2 * x\n", {"metadata": {"tests": ["assert f(2) == 4"]}}) == 1.0
