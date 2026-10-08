import json
from huggingface_hub import hf_hub_download
R = "mistralai/Mistral-7B-Instruct-v0.2"
idx = json.load(open(hf_hub_download(R, "model.safetensors.index.json", local_dir="F:/mistral")))
need = {f for k, f in idx["weight_map"].items()
        if any(f"layers.{l}.self_attn.{m}_proj.weight" in k for l in (0, 8, 16, 24, 31) for m in "qkv")}
print("shards:", sorted(need), flush=True)
for f in sorted(need):
    hf_hub_download(R, f, local_dir="F:/mistral"); print("done", f, flush=True)
