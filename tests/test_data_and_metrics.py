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


def test_gold_targets_score_perfectly(all_tasks, tmp_path):
    """Every task's own gold target must achieve metric 1.0 (metric/generator consistency)."""
    for task in all_tasks:
        t = _shrink(task)
        generate_dataset(t, tmp_path)
        metric = get_metric(t.metric)
        for ex in load_split(t, "test", tmp_path):
            assert metric(ex["target"], ex) == 1.0, (t.task_id, ex)


def test_splits_disjoint_and_schema(all_tasks, tmp_path):
    for task in all_tasks:
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


def test_committed_manifests_match_generators(all_tasks):
    """The committed manifests in data/manifests must be reproducible from the code."""
    for t in all_tasks:
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
