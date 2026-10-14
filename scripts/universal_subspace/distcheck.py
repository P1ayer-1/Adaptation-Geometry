import json, re, numpy as np
from safetensors import safe_open
info = json.load(open("subspace_N100.json")); lab = np.array(info["labels"]); big = np.bincount(lab).argmax()
key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
G = np.stack([safe_open(f, "np").get_tensor(key) for f, l in zip(info["files"], lab) if l == big])
T = G.mean(0).ravel()
k = lambda x: ((x - x.mean())**4).mean() / x.var()**2
print(f"group mean: max|x| {np.abs(T).max():.5f} (uniform bound 1/64 = {1/64:.5f}), std {T.std():.5f} (uniform {1/64/3**.5:.5f}), kurtosis {k(T):.2f} (uniform 1.80, gaussian 3.00)")
print("frac |x| > 1/64:", np.mean(np.abs(T) > 1/64))
print("group members:", [re.search(r"task\d+", f).group() for f, l in zip(info["files"], lab) if l == big][:40])
