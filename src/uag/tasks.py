"""Deterministic, rule-based generators for the ten Stage-0 transformations (spec §5.2).

Every generator is a pure function of a ``random.Random`` instance, so a dataset is fully
determined by (generator name, generator version, seed, kwargs). Targets are defined by
rules, never by another model's preferences. Instructions are deliberately minimal so that
the *behaviour* (style, format, label space) is carried by the adapter rather than spelled
out in the prompt.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Callable

GeneratorFn = Callable[..., dict[str, Any]]


@dataclass
class Generator:
    name: str
    fn: GeneratorFn
    instruction: str
    version: str


GENERATORS: dict[str, Generator] = {}


def register(name: str, instruction: str, version: str = "1"):
    def deco(fn: GeneratorFn) -> GeneratorFn:
        GENERATORS[name] = Generator(name, fn, instruction, version)
        return fn

    return deco


def get_generator(name: str) -> Generator:
    try:
        return GENERATORS[name]
    except KeyError as e:
        raise KeyError(f"unknown generator {name!r}; known: {sorted(GENERATORS)}") from e


# ---------------------------------------------------------------------------
# Shared vocabularies
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Bruno", "Chen", "Dara", "Elena", "Farid", "Grace", "Hugo", "Ines", "Jonas",
    "Kemal", "Lena", "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq",
    "Uma", "Victor", "Wen", "Ximena", "Yusuf", "Zoe",
]
LAST_NAMES = [
    "Adams", "Baker", "Costa", "Diaz", "Evans", "Fischer", "Garcia", "Hansen", "Ito", "Jensen",
    "Kowalski", "Larsen", "Moreau", "Novak", "Okafor", "Petrov", "Rossi", "Silva", "Tanaka",
    "Weber",
]
CITIES = [
    "Paris", "Lisbon", "Nairobi", "Osaka", "Toronto", "Lima", "Oslo", "Cairo", "Dublin",
    "Hanoi", "Denver", "Porto", "Seoul", "Quito", "Zurich", "Perth",
]
JOBS = [
    "nurse", "teacher", "pilot", "chef", "lawyer", "baker", "farmer", "dentist", "architect",
    "plumber", "librarian", "journalist", "engineer", "pharmacist",
]
ITEMS = ["apple", "book", "pencil", "marble", "ticket", "cookie", "stamp", "shell", "coin", "card"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PLACES = ["market", "library", "station", "museum", "bakery", "harbor"]

COUNTRY_CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
    ("Norway", "Oslo"), ("Egypt", "Cairo"), ("Ireland", "Dublin"), ("Vietnam", "Hanoi"),
    ("Portugal", "Lisbon"), ("Canada", "Ottawa"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Germany", "Berlin"), ("Poland", "Warsaw"), ("Greece", "Athens"), ("Chile", "Santiago"),
    ("Cuba", "Havana"), ("India", "New Delhi"), ("Argentina", "Buenos Aires"),
    ("Austria", "Vienna"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
    ("Hungary", "Budapest"), ("Thailand", "Bangkok"), ("Morocco", "Rabat"),
    ("Ghana", "Accra"), ("Colombia", "Bogota"), ("Turkey", "Ankara"), ("Mexico", "Mexico City"),
    ("Belgium", "Brussels"), ("Denmark", "Copenhagen"), ("Nepal", "Kathmandu"),
]


def _pick(rng: random.Random, xs):
    return xs[rng.randrange(len(xs))]


def _meta(**kw) -> dict[str, Any]:
    return kw


# ---------------------------------------------------------------------------
# T1 Sentiment classification
# ---------------------------------------------------------------------------

_SENT_SUBJECTS = ["The movie", "The restaurant", "This phone", "The hotel", "The concert",
                  "This book", "The service", "The new update", "The coffee", "The tour"]
_POS = ["wonderful", "delightful", "excellent", "charming", "superb", "enjoyable", "brilliant",
        "reliable", "pleasant", "fantastic"]
_NEG = ["terrible", "disappointing", "awful", "boring", "dreadful", "unpleasant", "mediocre",
        "unreliable", "frustrating", "annoying"]
_INTENS = ["", "really ", "truly ", "quite ", "absolutely "]
_TAILS = ["", " and I would say so again", " overall", " from start to finish",
          " according to everyone I asked"]


@register("sentiment", instruction="Classify the review.")
def gen_sentiment(rng: random.Random) -> dict[str, Any]:
    polarity = rng.random() < 0.5
    negate = rng.random() < 0.3
    adj = _pick(rng, _POS if polarity else _NEG)
    subj = _pick(rng, _SENT_SUBJECTS)
    verb = "was not" if negate else "was"
    text = f"{subj} {verb} {_pick(rng, _INTENS)}{adj}{_pick(rng, _TAILS)}."
    label = "positive" if (polarity != negate) else "negative"
    return {"input": text, "target": label,
            "metadata": _meta(choices=["positive", "negative"], label=label,
                              difficulty="negation" if negate else "plain")}


# ---------------------------------------------------------------------------
# T2 Natural-language inference
# ---------------------------------------------------------------------------

_PETS = ["a cat", "a dog", "two parrots", "a rabbit", "a goldfish"]


@register("nli", instruction="Does the premise entail the hypothesis?")
def gen_nli(rng: random.Random) -> dict[str, Any]:
    name = _pick(rng, FIRST_NAMES)
    job = _pick(rng, JOBS)
    city = _pick(rng, CITIES)
    premise = f"{name} is a {job} who lives in {city}."
    label = _pick(rng, ["entailment", "neutral", "contradiction"])
    if label == "entailment":
        hyp = _pick(rng, [f"{name} lives in {city}.", f"{name} works as a {job}.",
                          f"Someone in {city} is a {job}."])
    elif label == "contradiction":
        other_city = _pick(rng, [c for c in CITIES if c != city])
        other_job = _pick(rng, [j for j in JOBS if j != job])
        hyp = _pick(rng, [f"{name} has never lived in {city}.", f"{name} lives only in {other_city}.",
                          f"{name} is not a {job}.", f"{name} works only as a {other_job}."])
    else:
        hyp = _pick(rng, [f"{name} owns {_pick(rng, _PETS)}.", f"{name} is {rng.randint(21, 70)} years old.",
                          f"{name} was born on a {_pick(rng, DAYS)}.", f"{name} likes to read."])
    return {"input": f"Premise: {premise}\nHypothesis: {hyp}", "target": label,
            "metadata": _meta(choices=["entailment", "neutral", "contradiction"], label=label,
                              difficulty="plain")}


# ---------------------------------------------------------------------------
# T3 Paraphrase / semantic equivalence
# ---------------------------------------------------------------------------


def _event(name, n, item, place, day, form):
    plural = item + ("s" if n != 1 else "")
    if form == 0:
        return f"{name} bought {n} {plural} at the {place} on {day}."
    if form == 1:
        return f"On {day}, {name} bought {n} {plural} at the {place}."
    if form == 2:
        return f"On {day}, {n} {plural} were bought by {name} at the {place}."
    return f"At the {place} on {day}, {name} purchased {n} {plural}."


@register("paraphrase", instruction="Do the two sentences mean the same thing?")
def gen_paraphrase(rng: random.Random) -> dict[str, Any]:
    name, n, item = _pick(rng, FIRST_NAMES), rng.randint(2, 12), _pick(rng, ITEMS)
    place, day = _pick(rng, PLACES), _pick(rng, DAYS)
    f1 = rng.randrange(4)
    f2 = _pick(rng, [f for f in range(4) if f != f1])
    s1 = _event(name, n, item, place, day, f1)
    same = rng.random() < 0.5
    change = "none"
    if same:
        s2 = _event(name, n, item, place, day, f2)
    else:
        change = _pick(rng, ["name", "count", "item", "day", "place"])
        n2, name2, item2, day2, place2 = n, name, item, day, place
        if change == "name":
            name2 = _pick(rng, [x for x in FIRST_NAMES if x != name])
        elif change == "count":
            n2 = _pick(rng, [x for x in range(2, 13) if x != n])
        elif change == "item":
            item2 = _pick(rng, [x for x in ITEMS if x != item])
        elif change == "day":
            day2 = _pick(rng, [x for x in DAYS if x != day])
        else:
            place2 = _pick(rng, [x for x in PLACES if x != place])
        s2 = _event(name2, n2, item2, place2, day2, f2)
    label = "yes" if same else "no"
    return {"input": f"Sentence 1: {s1}\nSentence 2: {s2}", "target": label,
            "metadata": _meta(choices=["yes", "no"], label=label, difficulty=change)}


# ---------------------------------------------------------------------------
# T4 Structured extraction to JSON
# ---------------------------------------------------------------------------


@register("json_extraction", instruction="Extract the record.")
def gen_json_extraction(rng: random.Random) -> dict[str, Any]:
    name = f"{_pick(rng, FIRST_NAMES)} {_pick(rng, LAST_NAMES)}"
    age, city, job = rng.randint(19, 79), _pick(rng, CITIES), _pick(rng, JOBS)
    templates = [
        f"{name}, {age}, is a {job} from {city}.",
        f"Meet {name}: a {age}-year-old {job} based in {city}.",
        f"{name} works as a {job} in {city} and is {age} years old.",
        f"In {city} lives {name}, a {job} aged {age}.",
    ]
    record = {"name": name, "age": age, "city": city, "occupation": job}
    return {"input": _pick(rng, templates), "target": json.dumps(record),
            "metadata": _meta(record=record, difficulty="plain")}


# ---------------------------------------------------------------------------
# T5 / T6 Concise vs verbose answer style (shared question distribution)
# ---------------------------------------------------------------------------


def _question(rng: random.Random) -> tuple[str, str, str]:
    """Return (question, answer, explanation-seed)."""
    if rng.random() < 0.5:
        country, capital = _pick(rng, COUNTRY_CAPITALS)
        q = _pick(rng, [f"What is the capital of {country}?", f"Which city is the capital of {country}?"])
        expl = (f"{capital} is the city where the national government of {country} is seated, "
                f"which is what makes it the capital")
        return q, capital, expl
    a, b = rng.randint(2, 60), rng.randint(2, 40)
    op = _pick(rng, ["plus", "minus", "times"])
    ans = {"plus": a + b, "minus": a - b, "times": a * b}[op]
    q = f"What is {a} {op} {b}?"
    sym = {"plus": "+", "minus": "-", "times": "*"}[op]
    expl = f"computing {a} {sym} {b} step by step gives {ans}, so that is the result of the expression"
    return q, str(ans), expl


@register("concise_answer", instruction="Answer the question.")
def gen_concise(rng: random.Random, max_words: int = 4) -> dict[str, Any]:
    q, ans, _ = _question(rng)
    return {"input": q, "target": ans,
            "metadata": _meta(answer=ans, max_words=max_words, difficulty="plain")}


_VERBOSE_OPENERS = ["Good question.", "Let me explain this carefully.", "Here is a full answer."]
_VERBOSE_CLOSERS = ["I hope this detailed explanation makes the reasoning clear.",
                    "In summary, that is the complete answer to the question you asked.",
                    "That is the answer, together with the reasoning behind it."]


@register("verbose_answer", instruction="Answer the question.")
def gen_verbose(rng: random.Random, min_words: int = 25, max_words: int = 80) -> dict[str, Any]:
    q, ans, expl = _question(rng)
    target = (f"{_pick(rng, _VERBOSE_OPENERS)} The answer is {ans}. To explain, {expl}. "
              f"{_pick(rng, _VERBOSE_CLOSERS)}")
    return {"input": q, "target": target,
            "metadata": _meta(answer=ans, min_words=min_words, max_words=max_words,
                              difficulty="plain")}


# ---------------------------------------------------------------------------
# T7 Python code generation (checked by unit tests)
# ---------------------------------------------------------------------------

_FN_NAMES = ["solve", "compute", "transform", "helper", "process", "calc", "apply_rule", "f"]


def _py_spec(rng: random.Random) -> tuple[str, str, list[str]]:
    fn = _pick(rng, _FN_NAMES)
    kind = rng.randrange(8)
    a = rng.randint(2, 99)
    small = rng.randint(2, 9)
    if kind == 0:
        return (f"Write a Python function `{fn}(x)` that returns x plus {a}.",
                f"def {fn}(x):\n    return x + {a}",
                [f"assert {fn}(1) == {1 + a}", f"assert {fn}(-3) == {-3 + a}"])
    if kind == 1:
        return (f"Write a Python function `{fn}(x)` that returns x multiplied by {a}.",
                f"def {fn}(x):\n    return x * {a}",
                [f"assert {fn}(2) == {2 * a}", f"assert {fn}(0) == 0"])
    if kind == 2:
        return (f"Write a Python function `{fn}(xs)` that returns the sum of the elements of xs "
                f"that are greater than {a}.",
                f"def {fn}(xs):\n    return sum(x for x in xs if x > {a})",
                [f"assert {fn}([1, {a}, {a + 1}, {a + 5}]) == {2 * a + 6}", f"assert {fn}([]) == 0"])
    if kind == 3:
        return (f"Write a Python function `{fn}(s)` that returns the string s reversed.",
                f"def {fn}(s):\n    return s[::-1]",
                [f"assert {fn}('abc') == 'cba'", f"assert {fn}('') == ''"])
    if kind == 4:
        return (f"Write a Python function `{fn}(s)` that returns the string s repeated {small} times.",
                f"def {fn}(s):\n    return s * {small}",
                [f"assert {fn}('ab') == {'ab' * small!r}", f"assert {fn}('') == ''"])
    if kind == 5:
        return (f"Write a Python function `{fn}(xs)` that returns the first {small} elements of the list xs.",
                f"def {fn}(xs):\n    return xs[:{small}]",
                [f"assert {fn}(list(range(20))) == {list(range(small))}", f"assert {fn}([]) == []"])
    if kind == 6:
        return (f"Write a Python function `{fn}(n)` that returns True if n is divisible by {a} "
                f"and False otherwise.",
                f"def {fn}(n):\n    return n % {a} == 0",
                [f"assert {fn}({a * 3}) is True", f"assert {fn}({a * 3 + 1}) is False"])
    c = _pick(rng, list("aeiou"))
    return (f"Write a Python function `{fn}(s)` that returns how many times the letter "
            f"'{c}' occurs in the string s.",
            f"def {fn}(s):\n    return s.count({c!r})",
            [f"assert {fn}({(c * 3 + 'xyz')!r}) == 3", f"assert {fn}('') == 0"])


@register("python_codegen", instruction="Write the code.")
def gen_python(rng: random.Random) -> dict[str, Any]:
    spec, code, tests = _py_spec(rng)
    return {"input": spec, "target": code,
            "metadata": _meta(tests=tests, difficulty="plain")}


# ---------------------------------------------------------------------------
# T8 Arithmetic word problems
# ---------------------------------------------------------------------------


@register("arithmetic_word", instruction="Solve the problem.")
def gen_arithmetic(rng: random.Random) -> dict[str, Any]:
    name, item = _pick(rng, FIRST_NAMES), _pick(rng, ITEMS)
    a, b = rng.randint(3, 40), rng.randint(2, 30)
    kind = rng.randrange(3)
    if kind == 0:
        c = rng.randint(1, a + b - 1)
        ans = a + b - c
        q = (f"{name} has {a} {item}s and gets {b} more. Then {name} gives away {c}. "
             f"How many {item}s does {name} have now?")
        work = f"{a} + {b} = {a + b}. {a + b} - {c} = {ans}."
    elif kind == 1:
        k = rng.randint(2, 6)
        ans = a * k
        q = f"Each box holds {a} {item}s. {name} has {k} boxes. How many {item}s are there in total?"
        work = f"{a} * {k} = {ans}."
    else:
        k = rng.randint(2, 6)
        total = a * k
        ans = a
        q = f"{name} shares {total} {item}s equally among {k} friends. How many {item}s does each friend get?"
        work = f"{total} / {k} = {ans}."
    return {"input": q, "target": f"{work} Answer: {ans}",
            "metadata": _meta(answer=ans, difficulty=["add_sub", "mul", "div"][kind])}


# ---------------------------------------------------------------------------
# T9 Domain terminology transformation (lay -> clinical)
# ---------------------------------------------------------------------------

LAY_TO_CLINICAL = [
    ("heart attack", "myocardial infarction"), ("high blood pressure", "hypertension"),
    ("stroke", "cerebrovascular accident"), ("nosebleed", "epistaxis"),
    ("shortness of breath", "dyspnea"), ("chest pain", "angina"), ("fainting", "syncope"),
    ("itching", "pruritus"), ("low blood sugar", "hypoglycemia"), ("fast heartbeat", "tachycardia"),
    ("slow heartbeat", "bradycardia"), ("headache", "cephalalgia"), ("bruise", "contusion"),
    ("swelling", "edema"), ("fever", "pyrexia"), ("double vision", "diplopia"),
    ("hair loss", "alopecia"), ("joint pain", "arthralgia"), ("muscle pain", "myalgia"),
    ("trouble swallowing", "dysphagia"), ("coughing up blood", "hemoptysis"),
    ("kidney stone", "nephrolithiasis"), ("bedsore", "decubitus ulcer"), ("sleeplessness", "insomnia"),
]
_T9_TEMPLATES = [
    "The patient reported {a} and {b}.",
    "After the fall, she developed {a}.",
    "He was admitted with {a}, later complicated by {b}.",
    "History is notable for {a}.",
    "Symptoms included {a}, {b} and mild {c}.",
]


@register("clinical_terminology", instruction="Rewrite the note.")
def gen_clinical(rng: random.Random) -> dict[str, Any]:
    tmpl = _pick(rng, _T9_TEMPLATES)
    pairs = rng.sample(LAY_TO_CLINICAL, 3)
    lay = {k: p[0] for k, p in zip("abc", pairs)}
    clin = {k: p[1] for k, p in zip("abc", pairs)}
    used = [list(p) for k, p in zip("abc", pairs) if "{" + k + "}" in tmpl]
    return {"input": tmpl.format(**lay), "target": tmpl.format(**clin),
            "metadata": _meta(terms=used, difficulty=str(len(used)))}


# ---------------------------------------------------------------------------
# T10 Instruction-following format constraint
# ---------------------------------------------------------------------------

_WORDS = ["apple", "river", "stone", "cloud", "tiger", "lamp", "violin", "maple", "copper",
          "harbor", "velvet", "comet", "garden", "ember", "falcon", "prism", "quartz", "willow"]


@register("format_constraint", instruction="List the items.")
def gen_format(rng: random.Random, min_items: int = 3, max_items: int = 6) -> dict[str, Any]:
    items = rng.sample(_WORDS, rng.randint(min_items, max_items))
    target = "\n".join(f"- {w.upper()}" for w in items) + "\nEND"
    return {"input": ", ".join(items), "target": target,
            "metadata": _meta(items=items, difficulty=str(len(items)))}


# ---------------------------------------------------------------------------
# Stage 3 factorial task: arithmetic (T8 factor) + concise answer (T5 factor)
# ---------------------------------------------------------------------------


@register("arithmetic_concise", instruction="Solve the problem.")
def gen_arithmetic_concise(rng: random.Random, max_words: int = 4) -> dict[str, Any]:
    ex = gen_arithmetic(rng)
    ans = ex["metadata"]["answer"]
    return {"input": ex["input"], "target": str(ans),
            "metadata": _meta(answer=ans, max_words=max_words, factors=["arithmetic", "concise"],
                              difficulty=ex["metadata"]["difficulty"])}
