"""Task metric plugins and the statistical toolkit (spec §5.5, §14, §15).

Per-example metric plugins map ``(prediction, example) -> score in [0, 1]``. Aggregates are
means over examples except where a plugin registers a custom aggregator (macro-F1).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

MetricFn = Callable[[str, dict[str, Any]], float]
METRICS: dict[str, MetricFn] = {}


def register_metric(name: str):
    def deco(fn: MetricFn) -> MetricFn:
        METRICS[name] = fn
        return fn

    return deco


def get_metric(name: str) -> MetricFn:
    try:
        return METRICS[name]
    except KeyError as e:
        raise KeyError(f"unknown metric {name!r}; known {sorted(METRICS)}") from e


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def first_line(text: str) -> str:
    return text.strip().split("\n", 1)[0].strip()


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .!?\"'")


def word_count(text: str) -> int:
    return len(text.split())


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def last_number(text: str) -> float | None:
    nums = _NUM.findall(text.replace(",", ""))
    return float(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------


@register_metric("accuracy")
def accuracy(pred: str, ex: dict[str, Any]) -> float:
    return float(normalize(first_line(pred)) == normalize(ex["target"]))


@register_metric("exact_match")
def exact_match(pred: str, ex: dict[str, Any]) -> float:
    return float(normalize(first_line(pred)) == normalize(ex["target"]))


@register_metric("exact_number")
def exact_number(pred: str, ex: dict[str, Any]) -> float:
    m = re.search(r"Answer:\s*(-?[\d,]+(?:\.\d+)?)", pred)
    val = float(m.group(1).replace(",", "")) if m else last_number(pred)
    return float(val is not None and abs(val - float(ex["metadata"]["answer"])) < 1e-9)


def _answer_present(pred: str, answer: str) -> bool:
    return re.search(r"(?<![\w-])" + re.escape(normalize(answer)) + r"(?![\w])", normalize(pred)) is not None


@register_metric("concise_success")
def concise_success(pred: str, ex: dict[str, Any]) -> float:
    """Correct answer AND at most ``max_words`` words (length-constrained task success)."""
    p = pred.strip()
    return float(_answer_present(p, ex["metadata"]["answer"]) and
                 word_count(p) <= ex["metadata"]["max_words"])


@register_metric("verbose_success")
def verbose_success(pred: str, ex: dict[str, Any]) -> float:
    """Correct answer AND length within the calibrated [min_words, max_words] window."""
    p, m = pred.strip(), ex["metadata"]
    return float(_answer_present(p, m["answer"]) and m["min_words"] <= word_count(p) <= m["max_words"])


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


@register_metric("json_field_f1")
def json_field_f1(pred: str, ex: dict[str, Any]) -> float:
    """Field-level F1 over (key, value) pairs; 0 if the output is not a schema-valid object."""
    gold = ex["metadata"]["record"]
    obj = parse_json_object(pred)
    if obj is None or set(obj) - set(gold):
        return 0.0
    gold_pairs = {(k, json.dumps(v)) for k, v in gold.items()}
    pred_pairs = {(k, json.dumps(v)) for k, v in obj.items()}
    tp = len(gold_pairs & pred_pairs)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred_pairs), tp / len(gold_pairs)
    return 2 * p * r / (p + r)


def extract_code(pred: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", pred, flags=re.S)
    code = m.group(1) if m else pred
    # Stop at a trailing natural-language line after the function body.
    lines, out = code.strip("\n").split("\n"), []
    for line in lines:
        if out and line and not line.startswith((" ", "\t", "def ", "import ", "from ", "#", "@")):
            break
        out.append(line)
    return "\n".join(out)


def run_python_tests(code: str, tests: Sequence[str], timeout: float = 5.0) -> bool:
    """Execute generated code + asserts in an isolated subprocess (``python -I``) with a timeout.

    This is minimal isolation, not a security sandbox: run evaluations in a disposable
    container when evaluating untrusted models.
    """
    program = code + "\n\n" + "\n".join(tests) + "\n"
    try:
        r = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True,
                           timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


@register_metric("unit_test_pass")
def unit_test_pass(pred: str, ex: dict[str, Any]) -> float:
    return float(run_python_tests(extract_code(pred), ex["metadata"]["tests"]))


@register_metric("format_constraint")
def format_constraint(pred: str, ex: dict[str, Any], strict: bool = True) -> float:
    """Constraints: one ``- ITEM`` line per input item, in order, uppercase, then ``END``."""
    items = [w.upper() for w in ex["metadata"]["items"]]
    lines = [ln.rstrip() for ln in pred.strip().split("\n")]
    checks = []
    body = lines[:-1] if lines and lines[-1] == "END" else lines
    checks.append(bool(lines) and lines[-1] == "END")
    checks.append(all(ln.startswith("- ") for ln in body) and len(body) > 0)
    checks.append(all(ln[2:].isupper() for ln in body if len(ln) > 2))
    checks.append([ln[2:] for ln in body] == items)
    return float(all(checks)) if strict else sum(checks) / len(checks)


def macro_f1(preds: Sequence[str], golds: Sequence[str]) -> float:
    labels = sorted(set(golds))
    f1s = []
    for lab in labels:
        tp = sum(p == lab and g == lab for p, g in zip(preds, golds))
        fp = sum(p == lab and g != lab for p, g in zip(preds, golds))
        fn = sum(p != lab and g == lab for p, g in zip(preds, golds))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


# ---------------------------------------------------------------------------
# Transfer statistic and statistics toolkit
# ---------------------------------------------------------------------------


@dataclass
class LiftResult:
    recovered_lift: float | None
    direct_lift: float
    transfer_lift: float
    eligible: bool
    capped: bool
    reason: str = ""


def recovered_lift(s_transfer: float, s_base: float, s_direct: float, min_direct_lift: float = 0.05,
                   cap: tuple[float, float] = (-1.0, 2.0), higher_is_better: bool = True) -> LiftResult:
    """RecoveredLift = (S_transfer - S_base) / (S_directLoRA - S_base) (spec §5.5).

    Cells where direct-LoRA lift is below ``min_direct_lift`` are flagged ineligible (the ratio
    is unstable there) and return ``None``. Ratios outside ``cap`` are clipped and flagged.
    """
    sign = 1.0 if higher_is_better else -1.0
    direct = sign * (s_direct - s_base)
    transfer = sign * (s_transfer - s_base)
    if direct < min_direct_lift:
        return LiftResult(None, direct, transfer, False, False,
                          f"direct lift {direct:.3f} < min {min_direct_lift}")
    ratio = transfer / direct
    capped = not (cap[0] <= ratio <= cap[1])
    return LiftResult(float(np.clip(ratio, *cap)), direct, transfer, True, capped)


def bootstrap_ci(scores: Sequence[float], n_boot: int = 2000, level: float = 0.95, seed: int = 0,
                 stat: Callable[[np.ndarray], float] = np.mean) -> tuple[float, float, float]:
    """Percentile bootstrap CI over examples. Returns (point, low, high)."""
    x = np.asarray(scores, dtype=float)
    if x.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.array([stat(x[i]) for i in idx])
    a = (1 - level) / 2
    return float(stat(x)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                          level: float = 0.95, seed: int = 0) -> dict[str, float]:
    """Paired resampling of mean(a - b) over identical units (examples or base×task cells)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs aligned arrays")
    d = a - b
    point, lo, hi = bootstrap_ci(d, n_boot=n_boot, level=level, seed=seed)
    p_le_zero = float(np.mean(np.random.default_rng(seed).choice(d, size=(n_boot, d.size)).mean(1) <= 0)) \
        if d.size else math.nan
    return {"diff": point, "low": lo, "high": hi, "p_le_zero": p_le_zero,
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(d.size)}


def seed_summary(values: Sequence[float]) -> dict[str, float]:
    """Seed-level variation, reported separately from example-level CIs (spec §15)."""
    x = np.asarray([v for v in values if v is not None and not math.isnan(v)], float)
    if x.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max()), "n": int(x.size)}
