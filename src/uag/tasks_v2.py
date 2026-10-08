"""Version-2 task generators: a harder Stage-0 panel.

The v1 dry run showed that 5 worked examples in the prompt already bring an untrained
0.5B-1B base to 0.98-1.00 on JSON extraction and arithmetic, and that sentiment sits at
0.90-0.95 zero-shot. The LoRA lift was output format, not skill. Each v2 generator is designed
so that (a) the few-shot base stays well below ceiling and (b) a rule has to be learned from
many examples, not read off a handful of demonstrations:

- T1 hidden-rule triage: the label is review polarity XOR a hidden split of 40 products.
- T2 relational NLI: transitive comparisons over two chains (entail / contradict / incomparable).
- T3 paraphrase: role swaps, voice changes, number words, before/after inversions.
- T4 JSON: nested employer, optional (null) fields, a distractor person, and two house
  conventions (a hidden occupation → sector code, languages as ISO codes); exact record match.
- T5/T6: questions over a short context (argmax / sum / difference), answer judged by the
  first candidate mentioned, so a listing of every candidate doesn't count.
- T7 Python: 2-3 chained list steps from a 19-step library (filters, maps, order-dependent steps
  such as running totals, neighbour differences, every-second-element, dedupe) plus a final
  aggregate, or a string pipeline; tests are computed from the reference implementation.
- T8 arithmetic: 3-5 steps with 3-digit numbers and an irrelevant distractor number.
- T9 clinical: compositional medical morphology (organ root × procedure/condition suffix).
- T10 format: sort, number, conditional upper-casing by length, and a TOTAL footer.

Every generator is a pure function of its ``random.Random`` instance; module-level tables
are fixed lists (no set iteration), so datasets are reproducible bit for bit.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .tasks import CITIES, DAYS, FIRST_NAMES, ITEMS, JOBS, LAST_NAMES, _meta, _pick, register

V2 = "2"


# ---------------------------------------------------------------------------
# T1 Hidden-rule review triage (replaces sentiment: keep / return)
# ---------------------------------------------------------------------------

_PRODUCTS = [
    "blender", "backpack", "kettle", "lamp", "headset", "jacket", "monitor", "keyboard", "tent",
    "toaster", "umbrella", "wallet", "watch", "speaker", "router", "scooter", "stroller", "vacuum",
    "printer", "camera", "drill", "helmet", "mattress", "microwave", "notebook", "pillow", "razor",
    "sofa", "suitcase", "sweater", "tablet", "thermos", "treadmill", "tripod", "charger", "desk",
    "fan", "grill", "heater", "mixer",
]
# The hidden rule: for these products the label is flipped. Fixed by a named seed, never by
# set iteration, so it is identical on every machine.
_FLIPPED = sorted(random.Random("uag-t1-hidden-groups-v2").sample(_PRODUCTS, len(_PRODUCTS) // 2))
_POS2 = ["wonderful", "excellent", "charming", "superb", "enjoyable", "brilliant", "reliable",
         "pleasant", "fantastic", "sturdy", "handy", "lovely"]
_NEG2 = ["terrible", "disappointing", "awful", "flimsy", "dreadful", "unpleasant", "mediocre",
         "unreliable", "frustrating", "annoying", "useless", "clunky"]
_INTENS2 = ["", "really ", "truly ", "quite ", "absolutely ", "rather "]
_TAILS2 = ["", " overall", " from day one", " after a month of use", " for the price",
           " according to my whole family"]
_T1_FORMS = ["The {p} was {neg}{i}{a}{t}.", "I found the {p} {neg}{i}{a}{t}.", "This {p} is {neg}{i}{a}{t}.",
             "Honestly, the {p} was {neg}{i}{a}{t}."]


@register("hidden_rule_review", instruction="Triage the review.", version=V2)
def gen_hidden_rule(rng: random.Random) -> dict[str, Any]:
    product = _pick(rng, _PRODUCTS)
    positive = rng.random() < 0.5
    negate = rng.random() < 0.3
    text = _pick(rng, _T1_FORMS).format(p=product, neg="not " if negate else "", i=_pick(rng, _INTENS2),
                                        a=_pick(rng, _POS2 if positive else _NEG2), t=_pick(rng, _TAILS2))
    effective_positive = positive != negate
    flipped = product in _FLIPPED
    label = "keep" if effective_positive != flipped else "return"
    return {"input": text, "target": label,
            "metadata": _meta(choices=["keep", "return"], label=label, product=product,
                              difficulty=("flipped" if flipped else "plain") + ("+negation" if negate else ""))}


# ---------------------------------------------------------------------------
# T2 Relational NLI (transitivity over two comparison chains)
# ---------------------------------------------------------------------------

_RELATIONS = [("taller", "shorter"), ("older", "younger"), ("faster", "slower"), ("heavier", "lighter")]


def _claim(x: str, y: str, rel: tuple[str, str], rng: random.Random) -> str:
    """A sentence meaning 'x ranks above y' in either phrasing."""
    return f"{x} is {rel[0]} than {y}." if rng.random() < 0.5 else f"{y} is {rel[1]} than {x}."


@register("nli", instruction="Does the premise entail the hypothesis?", version=V2)
def gen_nli_v2(rng: random.Random) -> dict[str, Any]:
    rel = _pick(rng, _RELATIONS)
    n = rng.randint(4, 6)
    names = rng.sample(FIRST_NAMES, n)
    k = rng.randint(2, n - 2)
    chains = [names[:k], names[k:]]  # each chain in descending rank
    facts = [_claim(c[i], c[i + 1], rel, rng) for c in chains for i in range(len(c) - 1)]
    rng.shuffle(facts)
    label = _pick(rng, ["entailment", "neutral", "contradiction"])
    if label == "neutral":
        x, y = _pick(rng, chains[0]), _pick(rng, chains[1])
        if rng.random() < 0.5:
            x, y = y, x
        hyp, steps = _claim(x, y, rel, rng), "incomparable"
    else:
        c = _pick(rng, chains)
        i, j = sorted(rng.sample(range(len(c)), 2))
        hi, lo = c[i], c[j]
        hyp = _claim(hi, lo, rel, rng) if label == "entailment" else _claim(lo, hi, rel, rng)
        steps = str(j - i)
    return {"input": f"Premise: {' '.join(facts)}\nHypothesis: {hyp}", "target": label,
            "metadata": _meta(choices=["entailment", "neutral", "contradiction"], label=label, difficulty=steps)}


# ---------------------------------------------------------------------------
# T3 Paraphrase (roles, voice, number words, temporal inversions)
# ---------------------------------------------------------------------------

_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
              "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
              "nineteen"]


def _num(n: int, rng: random.Random) -> str:
    return _NUM_WORDS[n] if rng.random() < 0.5 else str(n)


def _transfer(a: str, b: str, n: int, item: str, day: str, form: int, rng: random.Random) -> str:
    things = f"{_num(n, rng)} {item}s"
    return [f"{a} gave {b} {things} on {day}.",
            f"On {day}, {a} gave {things} to {b}.",
            f"{b} received {things} from {a} on {day}.",
            f"On {day}, {things} were given to {b} by {a}.",
            f"{b} was given {things} by {a} on {day}."][form]


def _order(first: str, second: str, verb: str, form: int) -> str:
    return [f"{first} {verb} before {second}.", f"{second} {verb} after {first}.",
            f"{first} {verb} earlier than {second}.", f"{second} {verb} later than {first}."][form]


@register("paraphrase", instruction="Do the two sentences mean the same thing?", version=V2)
def gen_paraphrase_v2(rng: random.Random) -> dict[str, Any]:
    same = rng.random() < 0.5
    change = "none"
    if rng.random() < 0.6:
        a, b = rng.sample(FIRST_NAMES, 2)
        n, item, day = rng.randint(2, 19), _pick(rng, ITEMS), _pick(rng, DAYS)
        f1, f2 = rng.sample(range(5), 2)
        s1 = _transfer(a, b, n, item, day, f1, rng)
        if not same:
            change = _pick(rng, ["roles", "roles", "count", "day", "item"])
            if change == "roles":
                a, b = b, a
            elif change == "count":
                n = _pick(rng, [x for x in (n - 2, n - 1, n + 1, n + 2) if 2 <= x <= 19])
            elif change == "day":
                day = _pick(rng, [d for d in DAYS if d != day])
            else:
                item = _pick(rng, [x for x in ITEMS if x != item])
        s2 = _transfer(a, b, n, item, day, f2, rng)
    else:
        first, second = rng.sample(FIRST_NAMES, 2)
        verb = _pick(rng, ["arrived", "left", "finished"])
        f1, f2 = rng.sample(range(4), 2)
        s1 = _order(first, second, verb, f1)
        if not same:
            change = _pick(rng, ["order", "order", "verb", "person"])
            if change == "order":
                first, second = second, first
            elif change == "verb":
                verb = _pick(rng, [v for v in ("arrived", "left", "finished") if v != verb])
            else:
                second = _pick(rng, [x for x in FIRST_NAMES if x not in (first, second)])
        s2 = _order(first, second, verb, f2)
    label = "yes" if same else "no"
    return {"input": f"Sentence 1: {s1}\nSentence 2: {s2}", "target": label,
            "metadata": _meta(choices=["yes", "no"], label=label, difficulty=change)}


# ---------------------------------------------------------------------------
# T4 Nested JSON extraction with optional fields and a distractor person
# ---------------------------------------------------------------------------

_COMPANIES = ["Northwind Labs", "Bluefin Logistics", "Cedar Health", "Orbit Foods", "Juniper Bank",
              "Granite Works", "Harbor Media", "Pioneer Motors", "Silverline Energy", "Maple Textiles",
              "Summit Clinic", "Atlas Robotics"]
_LANGUAGES = ["English", "Spanish", "Portuguese", "Hindi", "Arabic", "French", "German", "Japanese",
              "Swahili", "Turkish", "Polish", "Korean"]
_ISO = {"English": "en", "Spanish": "es", "Portuguese": "pt", "Hindi": "hi", "Arabic": "ar", "French": "fr",
        "German": "de", "Japanese": "ja", "Swahili": "sw", "Turkish": "tr", "Polish": "pl", "Korean": "ko"}
# Hidden house convention: each occupation belongs to one of four arbitrary sector codes. Five
# demonstrations cannot reveal the mapping for all 14 occupations; training data can.
_SECTOR_ORDER = random.Random("uag-t4-sectors-v2").sample(JOBS, len(JOBS))
_SECTOR = {job: f"S{i % 4 + 1}" for i, job in enumerate(_SECTOR_ORDER)}


def _a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def _and_list(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


@register("json_extraction", instruction="Extract the record.", version=V2)
def gen_json_v2(rng: random.Random) -> dict[str, Any]:
    first, last = _pick(rng, FIRST_NAMES), _pick(rng, LAST_NAMES)
    name = f"{first} {last}"
    age, job, city, company = rng.randint(21, 69), _pick(rng, JOBS), _pick(rng, CITIES), _pick(rng, _COMPANIES)
    langs = rng.sample(_LANGUAGES, rng.randint(1, 3))
    manager = None
    if rng.random() < 0.6:
        manager = f"{_pick(rng, [x for x in FIRST_NAMES if x != first])} {_pick(rng, LAST_NAMES)}"
    start_year = rng.randint(1995, 2023) if rng.random() < 0.5 else None
    intro = _pick(rng, [f"{name}, {age}, works as {_a(job)} at {company} in {city}.",
                        f"Meet {name}: a {age}-year-old {job} employed by {company} ({city}).",
                        f"{name} ({age}) is {_a(job)} with {company}, based in {city}."])
    rest = [f"They speak {_and_list(langs)}."]
    if manager:
        rest.append(f"They report to {manager}.")
    if start_year:
        rest.append(f"They joined in {start_year}.")
    distractor = rng.random() < 0.7
    if distractor:
        d_first = _pick(rng, [x for x in FIRST_NAMES if x != first])
        d = f"{d_first} {_pick(rng, LAST_NAMES)}"
        rest.append(_pick(rng, [
            f"They often work with {d}, {_a(_pick(rng, JOBS))} aged {rng.randint(21, 69)} from {_pick(rng, CITIES)}.",
            f"Their friend {d} ({rng.randint(21, 69)}) is {_a(_pick(rng, JOBS))} in {_pick(rng, CITIES)}."]))
    rng.shuffle(rest)
    record = {"name": name, "age": age, "occupation": job, "sector": _SECTOR[job],
              "employer": {"name": company, "city": city}, "languages": [_ISO[x] for x in langs],
              "manager": manager, "start_year": start_year}
    return {"input": " ".join([intro, *rest]), "target": json.dumps(record),
            "metadata": _meta(record=record, difficulty="distractor" if distractor else "plain")}


# ---------------------------------------------------------------------------
# T5 / T6 Concise vs verbose answers to questions over a short context
# ---------------------------------------------------------------------------


def _question_v2(rng: random.Random) -> tuple[str, str, str, list[str] | None]:
    """Return (input, answer, explanation, candidate names or None for a numeric answer)."""
    people = rng.sample(FIRST_NAMES, rng.randint(3, 4))
    item = _pick(rng, ITEMS)
    counts = rng.sample(range(3, 61), len(people))
    context = "Context: " + _and_list([f"{p} has {c} {item}s" for p, c in zip(people, counts)]) + "."
    kind = rng.randrange(4)
    by = dict(zip(people, counts))
    if kind in (0, 1):
        best = max(people, key=by.get) if kind == 0 else min(people, key=by.get)
        word = "most" if kind == 0 else "fewest"
        q = f"Who has the {word} {item}s?"
        others = ", ".join(f"{by[p]}" for p in people if p != best)
        expl = (f"comparing all the counts, {best} has {by[best]} {item}s while the others have {others}, "
                f"so nobody has {'more' if kind == 0 else 'fewer'}")
        return f"{context}\nQuestion: {q}", best, expl, people
    x, y = rng.sample(people, 2)
    if kind == 2:
        ans = by[x] + by[y]
        q = f"How many {item}s do {x} and {y} have together?"
        expl = f"{x} has {by[x]} and {y} has {by[y]}, and adding {by[x]} and {by[y]} gives {ans} in total"
    else:
        if by[x] < by[y]:
            x, y = y, x
        ans = by[x] - by[y]
        q = f"How many more {item}s does {x} have than {y}?"
        expl = f"{x} has {by[x]} and {y} has {by[y]}, and subtracting {by[y]} from {by[x]} leaves {ans}"
    return f"{context}\nQuestion: {q}", str(ans), expl, None


@register("concise_answer", instruction="Answer the question.", version=V2)
def gen_concise_v2(rng: random.Random, max_words: int = 4) -> dict[str, Any]:
    inp, ans, _, cands = _question_v2(rng)
    return {"input": inp, "target": ans,
            "metadata": _meta(answer=ans, candidates=cands, max_words=max_words,
                              difficulty="name" if cands else "number")}


_VERBOSE_OPENERS2 = ["Good question.", "Let me explain this carefully.", "Here is a full answer."]
_VERBOSE_CLOSERS2 = ["I hope this detailed explanation makes the reasoning clear.",
                     "In summary, that is the complete answer to the question you asked.",
                     "That is the answer, together with the reasoning behind it."]


@register("verbose_answer", instruction="Answer the question.", version=V2)
def gen_verbose_v2(rng: random.Random, min_words: int = 25, max_words: int = 80) -> dict[str, Any]:
    inp, ans, expl, cands = _question_v2(rng)
    target = (f"{_pick(rng, _VERBOSE_OPENERS2)} The answer is {ans}. To explain, {expl}. "
              f"{_pick(rng, _VERBOSE_CLOSERS2)}")
    return {"input": inp, "target": target,
            "metadata": _meta(answer=ans, candidates=cands, min_words=min_words, max_words=max_words,
                              difficulty="name" if cands else "number")}


# ---------------------------------------------------------------------------
# T7 Python: composed pipelines, tests computed from the reference implementation
# ---------------------------------------------------------------------------

_FN_NAMES2 = ["solve", "compute", "transform", "pipeline", "process", "calc", "apply_rule", "f"]


def _list_step(rng: random.Random) -> tuple[str, str]:
    """One list -> list step as (description, code expression over ys)."""
    a, k, b, m, n = rng.randint(-5, 25), rng.randint(2, 7), rng.randint(1, 20), rng.randint(2, 6), rng.randint(2, 4)
    return _pick(rng, [
        (f"keep only the elements greater than {a}", f"[x for x in ys if x > {a}]"),
        (f"keep only the elements less than {a}", f"[x for x in ys if x < {a}]"),
        ("keep only the even elements", "[x for x in ys if x % 2 == 0]"),
        ("keep only the odd elements", "[x for x in ys if x % 2 == 1]"),
        (f"keep only the elements divisible by {k}", f"[x for x in ys if x % {k} == 0]"),
        ("square every element", "[x * x for x in ys]"),
        (f"multiply every element by {m}", f"[x * {m} for x in ys]"),
        (f"add {b} to every element", f"[x + {b} for x in ys]"),
        ("replace every element by its absolute value", "[abs(x) for x in ys]"),
        ("double the even elements and negate the odd ones", "[x * 2 if x % 2 == 0 else -x for x in ys]"),
        ("keep every second element, starting with the first", "ys[::2]"),
        ("keep every second element, starting with the second", "ys[1::2]"),
        ("reverse the order", "ys[::-1]"),
        ("replace each element by the running total up to and including it",
         "[sum(ys[:i + 1]) for i in range(len(ys))]"),
        ("replace the list by the differences between neighbouring elements (later minus earlier)",
         "[q - p for p, q in zip(ys, ys[1:])]"),
        ("remove repeated values, keeping the first occurrence", "list(dict.fromkeys(ys))"),
        ("sort in ascending order", "sorted(ys)"),
        (f"keep only the first {n} elements", f"ys[:{n}]"),
        (f"drop the first {n} elements", f"ys[{n}:]"),
    ])


def _list_spec(rng: random.Random, fn: str) -> tuple[str, str, Any]:
    steps = [_list_step(rng) for _ in range(rng.randint(2, 3))]
    t = rng.randint(0, 30)
    agg = _pick(rng, [("the sum of the result", "sum(ys)"), ("the number of elements in the result", "len(ys)"),
                      ("the largest element of the result, or 0 if it is empty", "max(ys) if ys else 0"),
                      ("the smallest element of the result, or 0 if it is empty", "min(ys) if ys else 0"),
                      ("the result itself", "ys"),
                      (f"how many elements of the result are greater than {t}", f"sum(1 for y in ys if y > {t})"),
                      ("the difference between the largest and the smallest element of the result, or 0 if it "
                       "is empty", "max(ys) - min(ys) if ys else 0")])
    words = ["first", "then", "then"]
    spec = (f"Write a Python function `{fn}(xs)` that takes a list of integers and, in this order, "
            + "; ".join(f"{words[i]} {d}" for i, (d, _) in enumerate(steps)) + f"; and returns {agg[0]}.")
    code = f"def {fn}(xs):\n    ys = list(xs)\n" + "".join(f"    ys = {c}\n" for _, c in steps) + f"    return {agg[1]}"
    inputs = [[rng.randint(-20, 40) for _ in range(rng.randint(4, 9))] for _ in range(3)] + [[]]
    return spec, code, inputs


_T7_WORDS = ["apple", "echo", "river", "ink", "stone", "orbit", "cloud", "umbra", "tiger", "atlas",
             "lamp", "igloo", "violin", "oak", "maple", "eagle", "copper", "ember", "falcon", "island"]


def _string_spec(rng: random.Random, fn: str) -> tuple[str, str, Any]:
    n = rng.randint(3, 5)
    filt = _pick(rng, [(f"longer than {n} letters", f"len(w) > {n}"), (f"shorter than {n} letters", f"len(w) < {n}"),
                       ("that start with a vowel", "w[0] in 'aeiou'"), ("that do not start with a vowel",
                                                                       "w[0] not in 'aeiou'")])
    tr = _pick(rng, [("reverses each of them", "w[::-1]"), ("upper-cases each of them", "w.upper()"),
                     ("keeps only the first letter of each of them", "w[0]"),
                     ("keeps only the last two letters of each of them", "w[-2:]"), ("leaves them unchanged", "w")])
    sep = _pick(rng, ["-", " ", ",", ""])
    spec = (f"Write a Python function `{fn}(s)` that splits the string s into words, keeps only the words "
            f"{filt[0]}, {tr[0]}, and returns them joined with {sep!r}.")
    code = (f"def {fn}(s):\n    ws = [w for w in s.split() if {filt[1]}]\n    ws = [{tr[1]} for w in ws]\n"
            f"    return {sep!r}.join(ws)")
    inputs = [" ".join(rng.choice(_T7_WORDS) for _ in range(rng.randint(3, 7))) for _ in range(3)] + [""]
    return spec, code, inputs


@register("python_codegen", instruction="Write the code.", version=V2)
def gen_python_v2(rng: random.Random) -> dict[str, Any]:
    fn = _pick(rng, _FN_NAMES2)
    family = "list" if rng.random() < 0.75 else "string"
    spec, code, inputs = (_list_spec if family == "list" else _string_spec)(rng, fn)
    ns: dict[str, Any] = {}
    exec(code, ns)  # our own reference implementation (deterministic, no model output involved)
    tests = [f"assert {fn}({x!r}) == {ns[fn](x)!r}" for x in inputs]
    return {"input": spec, "target": code, "metadata": _meta(tests=tests, difficulty=family)}


# ---------------------------------------------------------------------------
# T8 Multi-step arithmetic with larger numbers and a distractor
# ---------------------------------------------------------------------------


@register("arithmetic_word", instruction="Solve the problem.", version=V2)
def gen_arithmetic_v2(rng: random.Random) -> dict[str, Any]:
    name, item = _pick(rng, FIRST_NAMES), _pick(rng, ITEMS)
    cur = rng.randint(120, 950)
    story = [f"{name}'s shop has {cur} {item}s."]
    work = []
    n_steps = rng.randint(3, 4)
    for _ in range(n_steps):
        op = _pick(rng, ["add", "sub", "ship"])
        if op == "add":
            b = rng.randint(25, 400)
            story.append(_pick(rng, [f"Then {name} buys {b} more.", f"A supplier delivers {b} more."]))
            work.append(f"{cur} + {b} = {cur + b}.")
            cur += b
        elif op == "sub" and cur > 40:
            c = rng.randint(10, cur - 20)
            story.append(_pick(rng, [f"Then {name} sells {c}.", f"Customers buy {c} of them."]))
            work.append(f"{cur} - {c} = {cur - c}.")
            cur -= c
        else:
            k, b = rng.randint(2, 9), rng.randint(12, 60)
            story.append(f"Then {k} shipments of {b} {item}s each arrive.")
            work.append(f"{k} * {b} = {k * b}. {cur} + {k * b} = {cur + k * b}.")
            cur += k * b
    divs = [d for d in range(2, 10) if cur % d == 0]
    if divs and rng.random() < 0.5:
        d = _pick(rng, divs)
        story.append(f"Finally, the {item}s are split equally among {d} stores.")
        question = f"How many {item}s does each store get?"
        work.append(f"{cur} / {d} = {cur // d}.")
        cur //= d
    else:
        question = f"How many {item}s does the shop have now?"
    if rng.random() < 0.5:
        story.insert(rng.randint(1, len(story)), _pick(rng, [
            f"The shop has been open for {rng.randint(3, 40)} years.",
            f"{name} employs {rng.randint(2, 30)} people.",
            f"The shop is {rng.randint(100, 900)} meters from the station."]))
    return {"input": " ".join(story) + " " + question, "target": " ".join(work) + f" Answer: {cur}",
            "metadata": _meta(answer=cur, difficulty=f"{len(work)}_steps")}


# ---------------------------------------------------------------------------
# T9 Compositional clinical terminology (organ root × suffix)
# ---------------------------------------------------------------------------

# (lay organ, root, allowed suffixes): only combinations that are established medical terms.
_ROOTS = [
    ("stomach", "gastr", ["itis", "algia", "megaly", "ectomy", "pathy", "otomy", "scopy", "plasty"]),
    ("liver", "hepat", ["itis", "algia", "megaly", "ectomy", "pathy", "otomy"]),
    ("kidney", "nephr", ["itis", "algia", "megaly", "ectomy", "pathy", "otomy", "scopy"]),
    ("heart", "cardi", ["itis", "algia", "megaly", "pathy", "otomy"]),
    ("joint", "arthr", ["itis", "algia", "ectomy", "pathy", "otomy", "scopy", "plasty"]),
    ("nerve", "neur", ["itis", "algia", "ectomy", "pathy", "otomy", "plasty"]),
    ("nose", "rhin", ["itis", "algia", "otomy", "scopy", "plasty"]),
    ("ear", "ot", ["itis", "algia", "scopy", "plasty"]),
    ("brain", "encephal", ["itis", "algia", "pathy"]),
    ("vein", "phleb", ["itis", "ectomy", "otomy", "pathy"]),
    ("tongue", "gloss", ["itis", "algia", "ectomy", "plasty"]),
    ("mouth", "stomat", ["itis", "algia", "plasty"]),
    ("lymph node", "lymphaden", ["itis", "ectomy", "pathy"]),
]
_SUFFIX_LAY = {"itis": "inflammation of the {o}", "algia": "pain in the {o}", "megaly": "enlargement of the {o}",
               "pathy": "disease of the {o}", "ectomy": "surgical removal of the {o}",
               "otomy": "surgical incision into the {o}", "scopy": "examination of the {o} with a scope",
               "plasty": "surgical repair of the {o}"}
_CONDITIONS = ("itis", "algia", "megaly", "pathy")


def clinical_term(root: str, suffix: str) -> str:
    """Join a root and suffix: add a linking 'o' before a consonant; merge a doubled vowel."""
    if suffix[0] not in "aeiou":
        return root + "o" + suffix
    if root[-1] == suffix[0]:
        return root + suffix[1:]
    return root + suffix


_TERMS = [(_SUFFIX_LAY[s].format(o=organ), clinical_term(root, s), s in _CONDITIONS)
          for organ, root, sufs in _ROOTS for s in sufs]
_COND_TERMS = [t for t in _TERMS if t[2]]
_PROC_TERMS = [t for t in _TERMS if not t[2]]
_T9V2_TEMPLATES = [  # c = condition slot, p = procedure slot
    ("The patient presented with {c1}.", 1, 0),
    ("Imaging confirmed {c1} and {c2}.", 2, 0),
    ("She was scheduled for {p1}.", 0, 1),
    ("He was admitted with {c1} and underwent {p1}.", 1, 1),
    ("History is notable for {c1}, {c2} and prior {p1}.", 2, 1),
    ("After {p1}, he developed {c1}.", 1, 1),
    ("The team recommended {p1} to treat {c1}.", 1, 1),
]


@register("clinical_terminology", instruction="Rewrite the note.", version=V2)
def gen_clinical_v2(rng: random.Random) -> dict[str, Any]:
    tmpl, n_c, n_p = _pick(rng, _T9V2_TEMPLATES)
    conds, procs = rng.sample(_COND_TERMS, n_c), rng.sample(_PROC_TERMS, n_p)
    lay = {**{f"c{i + 1}": t[0] for i, t in enumerate(conds)}, **{f"p{i + 1}": t[0] for i, t in enumerate(procs)}}
    clin = {**{f"c{i + 1}": t[1] for i, t in enumerate(conds)}, **{f"p{i + 1}": t[1] for i, t in enumerate(procs)}}
    return {"input": tmpl.format(**lay), "target": tmpl.format(**clin),
            "metadata": _meta(terms=[[t[0], t[1]] for t in conds + procs], difficulty=str(n_c + n_p))}


# ---------------------------------------------------------------------------
# T10 Format: sorted, numbered, conditional upper-casing, TOTAL footer
# ---------------------------------------------------------------------------

_WORDS2 = ["apple", "river", "stone", "cloud", "tiger", "lamp", "violin", "maple", "copper", "harbor",
           "velvet", "comet", "garden", "ember", "falcon", "prism", "quartz", "willow", "ash", "beacon",
           "canyon", "delta", "fern", "glacier", "hazel", "iris", "jade", "kite", "lagoon", "meadow",
           "nectar", "olive", "pepper", "raven", "saffron", "thistle", "urchin", "walnut", "yarrow", "zephyr"]


def format_v2_lines(items: list[str]) -> list[str]:
    """The rule: alphabetical order; 'N. item'; items longer than 5 letters in UPPER case,
    the rest in lower case; then 'TOTAL: n'."""
    lines = [f"{i + 1}. {w.upper() if len(w) > 5 else w.lower()}" for i, w in enumerate(sorted(items))]
    return lines + [f"TOTAL: {len(items)}"]


@register("format_constraint", instruction="List the items.", version=V2)
def gen_format_v2(rng: random.Random, min_items: int = 4, max_items: int = 7) -> dict[str, Any]:
    items = rng.sample(_WORDS2, rng.randint(min_items, max_items))
    return {"input": ", ".join(items), "target": "\n".join(format_v2_lines(items)),
            "metadata": _meta(items=items, difficulty=str(len(items)))}
