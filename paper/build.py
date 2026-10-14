"""Build the arXiv LaTeX source and PDF of docs/note_lora_universal_subspace.md.

  1. Split the note into title, author block, date, abstract and body.
  2. Turn the plain-Unicode formulas into LaTeX maths (each replacement must match exactly once,
     or as many times as stated, so an edit to the note cannot silently skip one).
  3. Point local links at the repository commit cited in the note; make Figure 1 a real figure.
  4. Convert with pandoc to a standalone main.tex, then replace every remaining non-ASCII
     character with a LaTeX command, so the source compiles with pdfLaTeX (arXiv) as well as XeTeX.
  5. Compile main.pdf with tectonic.

usage: python paper/build.py   (needs pandoc and tectonic, e.g. the micromamba env "texbuild")
"""
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NOTE = ROOT / "docs" / "note_lora_universal_subspace.md"
BIN = Path.home() / "micromamba" / "envs" / "texbuild" / "bin"
PANDOC = shutil.which("pandoc") or str(BIN / "pandoc")
TECTONIC = shutil.which("tectonic") or str(BIN / "tectonic")
COMMIT = "35296cb03172d6fee33797979a4e4e5d03e90318"
REPO = "https://github.com/P1ayer-1/Adaptation-Geometry"

src = NOTE.read_text()

# ---------------------------------------------------------------- 1. split
title = re.match(r"# (.+)\n", src).group(1)
abstract = src[src.index("## Abstract\n") + len("## Abstract\n"):src.index("## 1. Introduction")].strip()
body = src[src.index("## 1. Introduction"):]
date = re.search(r"^\*(\w+ \d+, \d{4})\*$", src, re.M).group(1)

# ---------------------------------------------------------------- 2. maths
IN, OUT = r"_{\text{in}}", r"_{\text{out}}"
MATH = [  # (plain text, LaTeX, expected count)
    ("ΔW = BA on the input side", r"$\Delta W = BA$ on the input side", 1),
    ('directly on ΔW"', r'directly on $\Delta W$"', 1),
    ('full update ΔW=BA"', r'full update $\Delta W = BA$"', 1),
    ("‖A − A₀‖_F / ‖A₀‖_F", r"$\|A - A_0\|_F / \|A_0\|_F$", 1),
    ("scaling α/r = 2", r"scaling $\alpha/r = 2$", 1),
    ("A is r × d_in (d_in = 4,096); B is\nd_out × r, with d_out = 4,096",
     rf"A is $r \times d{IN}$ ($d{IN} = 4{{,}}096$); B is\n$d{OUT} \times r$, with $d{OUT} = 4{{,}}096$", 1),
    ("(N·r) × d matrix", r"$(N r) \times d$ matrix", 1),
    ("500r × d matrix", r"$500r \times d$ matrix", 1),
    ("Writing ΔW_i = B_i A_i, we use the uncentred operators\nC_in = Σᵢ ΔW_iᵀ ΔW_i and C_out = Σᵢ ΔW_i ΔW_iᵀ, whose spectra do not depend on how ΔW_i is split",
     rf"Writing $\Delta W_i = B_i A_i$, we use the uncentred operators\n$C{IN} = \sum_i \Delta W_i^\top \Delta W_i$ and $C{OUT} = \sum_i \Delta W_i \Delta W_i^\top$, whose spectra do not depend on how $\Delta W_i$ is split", 1),
    ("QR of B and Aᵀ", r"QR of $B$ and $A^\top$", 1),
    ("(every ΔW_i scaled to ‖ΔW_i‖_F = 1)", r"(every $\Delta W_i$ scaled to $\|\Delta W_i\|_F = 1$)", 1),
    ("U(−1/64, 1/64)", r"$U(-1/64, 1/64)$", 1),
    ("Σᵢ‖Aᵢ − m‖² ≤ Σᵢ‖Aᵢ − c‖²", r"$\sum_i \|A_i - m\|^2 \le \sum_i \|A_i - c\|^2$", 1),
    ("(ΔW′ = U Uᵀ B A V Vᵀ)", r"($\Delta W' = U U^\top B A V V^\top$)", 1),
    ("(ΔW′ = B A V Vᵀ,", r"($\Delta W' = B A V V^\top$,", 1),
    ("(ΔW′ = U Uᵀ B A,", r"($\Delta W' = U U^\top B A$,", 1),
    ("C_in = Σ ΔWᵀΔW and C_out = Σ ΔW ΔWᵀ", rf"$C{IN} = \sum \Delta W^\top \Delta W$ and $C{OUT} = \sum \Delta W \Delta W^\top$", 1),
    ("‖ΔW‖²_F", r"$\|\Delta W\|_F^2$", 2),
    ("p ≤ 3 × 10⁻⁵", r"$p \le 3 \times 10^{-5}$", 1),
    ("mean cos² of", r"mean $\cos^2$ of", 1),
    ("chance is\n16/d.", "chance is\n$16/d$.", 1),
    ("chance level k/d.", "chance level $k/d$.", 1),
    ("U Σᵢ Vᵀ", r"$U \Sigma_i V^\top$", 1),
    ("n≥100", r"$n \geq 100$", 1),
    ("1 − ‖UᵀΔW_iV‖²_F", r"$1 - \|U^\top \Delta W_i V\|_F^2$", 1),
    ("Σᵢ‖UΣᵢVᵀ − ΔW_i‖²_F / Σᵢ‖ΔW_i‖²_F",
     r"$\sum_i \|U \Sigma_i V^\top - \Delta W_i\|_F^2 / \sum_i \|\Delta W_i\|_F^2$", 1),
    ("(uncentred C_in, C_out)", rf"(uncentred $C{IN}$, $C{OUT}$)", 1),
    ("2 × 10⁻⁴", r"$2 \times 10^{-4}$", 1),
]
for a, b, n in MATH:
    b = b.replace("\\n", "\n")  # raw strings above write line breaks as a literal backslash-n
    c = body.count(a) + abstract.count(a)
    assert c == n, f"expected {n} x {a!r}, found {c}"
    body, abstract = body.replace(a, b), abstract.replace(a, b)
GENERIC = [  # leftovers, applied after the exact replacements
    ("ΔW_i", r"$\Delta W_i$"), ("ΔW", r"$\Delta W$"), ("C_in", rf"$C{IN}$"), ("C_out", rf"$C{OUT}$"),
    ("d_in", rf"$d{IN}$"), ("d_out", rf"$d{OUT}$"),
]
for a, b in GENERIC:
    body, abstract = body.replace(a, b), abstract.replace(a, b)
left = sorted(set(re.findall(r"[ΔΣ‖ᵀᵢ₀′]", body + abstract)))
assert not left, f"unconverted formula characters: {left}"

# ---------------------------------------------------------------- 3. links and figure
body = body.replace("[`dev_scale_report.md`](dev_scale_report.md)",
                    f"[`dev_scale_report.md`]({REPO}/blob/{COMMIT}/docs/dev_scale_report.md)")
fig = re.search(r"!\[Figure 1\]\(figures/figure1_spectrum\.png\)\n\n\*Figure 1\. (.+?)\*\n", body, re.S)
assert fig, "Figure 1 block not found"
caption = " ".join(fig.group(1).split())
body = body.replace(fig.group(0), f"![{caption}](figures/figure1_spectrum.pdf){{width=80%}}\n")
local = re.findall(r"\]\((?!https?://|figures/)[^)]+\)", body)
assert not local, f"local links left: {local}"

# ---------------------------------------------------------------- 3b. ranges and table widths
# numeric ranges ("0.04-0.22", "8-31") get an en dash; model names and dates are left alone
RANGE = re.compile(r"(?<![\w.\-])(\d[\d,]*(?:\.\d+)?)-(\d[\d,]*(?:\.\d+)?)(?![\w\-]|\.\d)")
body, abstract = RANGE.sub(r"\1–\2", body), RANGE.sub(r"\1–\2", abstract)


def widen(block):
    """Set each pipe-table column's relative width from its longest body cell or header word."""
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in block]
    head, rest = rows[0], rows[2:]
    w = []
    for j in range(len(head)):
        cells = [r[j] for r in rest if j < len(r)]
        longest_word = max((len(x) for x in head[j].split()), default=3)
        width = max([longest_word, 4] + [len(c) for c in cells])
        if any(re.search(r"[A-Za-z]{3}", c) for c in cells):  # text column: may wrap, so cap its share
            width = min(width, max(longest_word, 14))
        w.append(width)
    block[1] = "|" + "|".join("-" * x for x in w) + "|"
    return block


lines, out, i = body.split("\n"), [], 0
while i < len(lines):
    if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-|: ]+\|$", lines[i + 1]):
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        out += widen(lines[i:j])
        i = j
    else:
        out.append(lines[i])
        i += 1
body = "\n".join(out)

# ---------------------------------------------------------------- 4. pandoc
build = HERE / "build"
build.mkdir(exist_ok=True)
(HERE / "figures").mkdir(exist_ok=True)
shutil.copy(ROOT / "docs" / "figures" / "figure1_spectrum.pdf", HERE / "figures" / "figure1_spectrum.pdf")
author = r"Noah Persaud\\ Holdout Labs\\ \texttt{noah@holdoutlabs.ai}"
meta = (f"---\ntitle: \"{title}\"\nauthor: '`{author}`{{=latex}}'\ndate: \"{date}\"\nabstract: |\n"
        + "\n".join("  " + l for l in abstract.splitlines()) + "\n---\n\n")
(build / "paper.md").write_text(meta + body)
header = r"""
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small\setlength{\tabcolsep}{4pt}}
\setlength{\emergencystretch}{3em}
\usepackage{amssymb}
"""
(build / "header.tex").write_text(header)
subprocess.run([PANDOC, str(build / "paper.md"), "-s", "-o", str(HERE / "main.tex"),
                "--from", "markdown+tex_math_dollars+raw_attribute", "--shift-heading-level-by=-1",
                "-V", "documentclass=article", "-V", "fontsize=11pt", "-V", "geometry:margin=1in",
                "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
                "-H", str(build / "header.tex")], check=True)

# ---------------------------------------------------------------- 4b. ASCII-only source
UNI = {"ü": r"\"{u}", "—": "---", "–": "--", "×": r"\texorpdfstring{\ensuremath{\times}}{x}", "−": r"\ensuremath{-}",
       "≤": r"\ensuremath{\le}", "≥": r"\ensuremath{\ge}", "±": r"\ensuremath{\pm}", "·": r"\ensuremath{\cdot}",
       "α": r"\ensuremath{\alpha}", "²": r"\ensuremath{^{2}}", "⁻": r"\ensuremath{^{-}}", "⁴": r"\ensuremath{^{4}}",
       "⁵": r"\ensuremath{^{5}}", "’": "'", "‘": "`", "“": "``", "”": "''", "é": r"\'{e}", "á": r"\'{a}"}
tex = (HERE / "main.tex").read_text()
for a, b in UNI.items():
    tex = tex.replace(a, b)
bad = sorted(set(c for c in tex if ord(c) > 127))
assert not bad, f"non-ASCII characters left in main.tex: {bad}"
# en-dash ranges inside tables must not break across lines
tex = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}",
             lambda m: re.sub(r"(\d)--(\d)", r"\1\\nobreakdash--\2", m.group(0)), tex, flags=re.S)
# let long file names in \texttt break after /, _ and -
tex = re.sub(r"\\texttt\{([^{}]*)\}",
             lambda m: r"\texttt{" + m.group(1).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}").replace("-", r"-\allowbreak{}") + "}",
             tex)
(HERE / "main.tex").write_text(tex)

# ---------------------------------------------------------------- 5. compile
subprocess.run([TECTONIC, "--keep-logs", str(HERE / "main.tex")], check=True, cwd=HERE)
log = (HERE / "main.log").read_text(errors="replace")
over = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)[^\n]*\n", log)
print("overfull boxes (pt):", [o for o in over if float(o) > 1.0])
print("wrote", HERE / "main.tex", "and", HERE / "main.pdf")
