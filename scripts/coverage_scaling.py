"""Coverage curve: how much of a held-out task's output-side update lies inside the output
directions of the first n map-training tasks (same base, seed 0), for n = 1..N.

    python scripts/coverage_scaling.py configs/experiments/dev_3080_frozenA_scale.yaml \
        --order T3_paraphrase T7_python T9_clinical T10_format T1_hidden_rule T2_nli T5_concise T6_verbose \
        --heldout T4_json T8_arithmetic

k=16 uses the top-16 directions of the stacked training updates (the maps' coordinate system);
k=all uses every direction they span (16 per task). A second seed of the same task is the
reference for "same task" when it exists.
"""
import argparse
import json

import numpy as np

from uag.config import load_experiment
from uag.extract_delta import DeltaSet
from uag.pipeline import Paths
from uag.transfer import _coordinate_basis


def coverage(held: DeltaSet, train: list[DeltaSet], k: int | None) -> float:
    vals = []
    for n, lr in held.modules.items():
        tr = [t.modules[n] for t in train]
        U = _coordinate_basis(tr, k or 16 * len(tr), "U")
        UL = lr.U * lr.S
        vals.append(float(((U.T @ UL) ** 2).sum() / (UL ** 2).sum()))
    return float(np.mean(vals))


ap = argparse.ArgumentParser()
ap.add_argument("experiment")
ap.add_argument("--order", nargs="+", required=True)
ap.add_argument("--heldout", nargs="+", required=True)
args = ap.parse_args()
exp = load_experiment(args.experiment)
paths = Paths(exp)
out = {"order": args.order, "heldout": args.heldout, "rows": []}
for base in exp.bases:
    ds = {t: DeltaSet.load(paths.run_dir(base.name, t, 0) / "spectral") for t in args.order + args.heldout}
    for h in args.heldout:
        line = []
        for n in range(1, len(args.order) + 1):
            tr = [ds[t] for t in args.order[:n]]
            row = {"base": base.name, "heldout": h, "n_train": n, "k16": coverage(ds[h], tr, 16),
                   "k_all": coverage(ds[h], tr, None)}
            out["rows"].append(row)
            line.append(f"n={n}: {row['k16']:.2f}/{row['k_all']:.2f}")
        print(f"{base.name:<18}{h:<15}" + "  ".join(line), flush=True)
f = paths.exp_results / "coverage_scaling.json"
f.parent.mkdir(parents=True, exist_ok=True)
f.write_text(json.dumps(out, indent=1))
print(f"(k=16 / k=all)  wrote {f}")
