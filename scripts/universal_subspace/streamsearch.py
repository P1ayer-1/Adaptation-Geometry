"""Slide the shared-group mean of layer-0 q_proj A (row 0) along each seed's torch CPU uniform stream.

Score = cosine between the target row and a 4096-long window of the stream (windows of iid
U(-1,1) have norm ~sqrt(4096/3)). Noise sd ~0.016, so anything above ~0.2 is a hit.
"""
import json, sys
import numpy as np, torch
from safetensors import safe_open

info = json.load(open(sys.argv[1]))
seeds = [int(s) for s in sys.argv[2].split(",")] if "," in sys.argv[2] else range(*map(int, sys.argv[2].split(":")))
M = 1 << int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 26
lab = np.array(info["labels"])
big = np.bincount(lab).argmax()
key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
T = np.mean([safe_open(f, "np").get_tensor(key) for f, l in zip(info["files"], lab) if l == big], 0)
if len(sys.argv) > 4:
    T = safe_open(sys.argv[4], "np").get_tensor(key)
t = torch.tensor(T[0], dtype=torch.float32)
t = t / t.norm()
L, C = t.numel(), 1 << 21
K = torch.fft.rfft(t.flip(0), n=C)
norm = (L / 3) ** 0.5
torch.set_num_threads(12)
for seed in seeds:
    torch.manual_seed(seed)
    u = torch.empty(M).uniform_(-1, 1)
    best, pos = 0.0, -1
    step = C - L + 1
    for s in range(0, M - L + 1, step):
        seg = u[s:s + C]
        if seg.numel() < L:
            break
        r = torch.fft.irfft(torch.fft.rfft(seg, n=C) * K, n=C)[L - 1:seg.numel()] / norm
        i = int(r.abs().argmax())
        if abs(float(r[i])) > abs(best):
            best, pos = float(r[i]), s + i
    flag = "  <== HIT" if abs(best) > 0.2 else ""
    print(f"seed {seed}: best cos {best:+.3f} at offset {pos}{flag}", flush=True)
