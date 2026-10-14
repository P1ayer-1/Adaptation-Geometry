"""Brute-force the torch seed behind the shared LoRA A init of the Lots-of-LoRAs collection.

Target: mean layer-0 q_proj A over the largest seed group (from subspace_N*.json).
PEFT (LoraLayer.update_layer) builds nn.Linear A, then nn.Linear B (each draws a kaiming
init), then reset_lora_parameters redraws A with kaiming_uniform_(a=sqrt(5)). All draws are
U(-1/sqrt(fan_in), +1/sqrt(fan_in)), so we test which draw (offset) of which seed matches.
"""
import json, math, sys
import numpy as np, torch
from safetensors import safe_open

info = json.load(open(sys.argv[1]))
lo, hi = int(sys.argv[2]), int(sys.argv[3])
lab = np.array(info["labels"])
big = np.bincount(lab).argmax()
files = [f for f, l in zip(info["files"], lab) if l == big]
key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
T = np.mean([safe_open(f, "np").get_tensor(key) for f in files], 0)
T = torch.tensor(T / np.linalg.norm(T)).flatten()
singles = [f for f, l in zip(info["files"], lab) if np.sum(lab == l) == 1][:20]
S = torch.stack([torch.tensor(safe_open(f, "np").get_tensor(key)).flatten() for f in singles])
S = S / S.norm(dim=1, keepdim=True)

best = []
for seed in range(lo, hi):
    torch.manual_seed(seed)
    seq = []
    for step in range(6):
        w = torch.empty(16, 4096)
        torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        seq.append(w.flatten())
    # also: A-draw, B-draw(16->4096), A-redraw  (PEFT order)
    torch.manual_seed(seed)
    a1 = torch.empty(16, 4096); torch.nn.init.kaiming_uniform_(a1, a=math.sqrt(5))
    b1 = torch.empty(4096, 16); torch.nn.init.kaiming_uniform_(b1, a=math.sqrt(5))
    a2 = torch.empty(16, 4096); torch.nn.init.kaiming_uniform_(a2, a=math.sqrt(5))
    cands = {f"A-draw#{i}": v for i, v in enumerate(seq)} | {"peft(A,B,A)": a2.flatten()}
    for name, v in cands.items():
        v = v / v.norm()
        c = float(v @ T)
        cs = float((S @ v).abs().max())
        if c > 0.3 or cs > 0.3:
            print(f"HIT seed {seed} {name}: cos to big group {c:.4f}, max cos to singletons {cs:.4f}", flush=True)
        best.append((c, seed, name))
best.sort(reverse=True)
print("top:", best[:5])
