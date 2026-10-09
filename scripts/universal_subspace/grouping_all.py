"""Inferred shared-initialisation groups across the whole Lots-of-LoRAs rank-16 Mistral collection.

usage: python grouping_all.py IDS.txt PAPER_MANIFEST.json OUT.json

Uses only layer-0 q_proj A per adapter. Reports group sizes, the shared fraction by task-number
range, and the shared fraction and paper-style top-16 share (layer-0 q A) for: the paper's set,
our earlier random 500, the lowest-numbered 500, and the whole collection.
"""
import json
import re
import sys

import numpy as np
import torch
from safetensors import safe_open

from common import PFX, components, cos_matrix, paper_spectrum, quantiles

ids_path, man_path, out = sys.argv[1:4]
ids = [l.strip() for l in open(ids_path) if l.strip()]
task = lambda i: int(re.search(r"task(\d+)$", i).group(1))
A = []
for i in ids:
    with safe_open(f"F:/lol/{i.split('/')[1]}/adapter_model.safetensors", "pt") as h:
        A.append(h.get_tensor(PFX.format(0, "q", "A")).double())
A = torch.stack(A)
C = cos_matrix(A)
lab = components(C, 0.5)
sizes = np.bincount(lab)
big = sizes.argmax()
is_shared = lab == big
iu = torch.triu_indices(len(ids), len(ids), 1)
same = torch.as_tensor(lab[iu[0]] == lab[iu[1]])
pair = C[iu[0], iu[1]]
res = dict(n=len(ids), groups=int(len(sizes)), largest=int(sizes.max()), sizes_desc=sorted(sizes.tolist(), reverse=True)[:10],
           within_pair_cos=quantiles(pair[same]), between_pair_cos=quantiles(pair[~same]))
bins = [(0, 200), (200, 400), (400, 600), (600, 1000), (1000, 1400), (1400, 2000)]
res["shared_fraction_by_task_number"] = {
    f"{a}-{b}": dict(n=int(sum(a <= task(i) < b for i in ids)),
                     shared=float(np.mean([s for i, s in zip(ids, is_shared) if a <= task(i) < b] or [np.nan])))
    for a, b in bins}
paper = {a["id"] for a in json.load(open(man_path))["adapters"] if a["role"] != "ood"}
sets = {"paper_v3_fit_set": [k for k, i in enumerate(ids) if i in paper],
        "earlier_random_500": list(range(500)),
        "lowest_numbered_500": sorted(range(len(ids)), key=lambda k: task(ids[k]))[:500],
        "whole_collection": list(range(len(ids)))}
for nm, ix in sets.items():
    ix = np.array(ix)
    res[nm] = dict(n=int(len(ix)), shared_fraction=float(is_shared[ix].mean()),
                   top16_L0q_A=paper_spectrum(A[ix].reshape(-1, A.shape[-1]))["top16"])
res["shared_ids"] = [i for i, s in zip(ids, is_shared) if s]
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "shared_ids"}, indent=1))
