"""Figure 1 of the note: cumulative variance of the paper-style A spectrum (layer 16 q_proj).

usage: python figure1.py SUPPLEMENT.json OUT_STEM   (writes OUT_STEM.png and OUT_STEM.pdf)
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sup, stem = sys.argv[1:3]
c = json.load(open(sup))["curves_L16q_A"]
n = c["n_matched"]
# Reference categorical palette, slots 1-5 in fixed order; dash patterns + the legend carry
# identity without colour.
series = [
    ("full_set", "Paper's adapter set (497)", "#2a78d6", "-"),
    ("no_training_sim", "Same seed grouping, no training", "#eb6834", (0, (5, 2))),
    ("shared_group", f"Shared-seed group ({n})", "#1baf7a", (0, (1, 1.5))),
    ("singletons", f"Own-seed adapters ({n})", "#eda100", (0, (6, 2, 1, 2))),
    ("independent_inits", f"Independent random inits ({n})", "#e87ba4", (0, (2, 2))),
]
ink, muted, grid = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
xmax = 128
for key, label, col, ls in series:
    y = c[key][:xmax]
    x = range(1, len(y) + 1)
    ax.plot(x, y, color=col, linestyle=ls, linewidth=2, label=label, solid_capstyle="round")
ax.axvline(16, color=muted, linewidth=1, linestyle=":")
ax.text(17.5, 0.965, "LoRA rank (16)", color=muted, fontsize=7.5)
ax.set_xlim(1, xmax)
ax.set_ylim(0, 1)
ax.set_xlabel("Number of principal components")
ax.set_ylabel("Cumulative share of variance")
ax.grid(axis="y", color=grid, linewidth=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.71), frameon=False, fontsize=7.5, labelcolor=ink,
          handlelength=3.2)
ax.set_title("LoRA A matrices, layer 16 q_proj: stacked rank vectors, centred (the paper's analysis)",
             fontsize=8.5, color=ink, loc="left")
fig.savefig(stem + ".png", bbox_inches="tight")
fig.savefig(stem + ".pdf", bbox_inches="tight")
print("wrote", stem)
