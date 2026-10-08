import sys, concurrent.futures as cf
from huggingface_hub import hf_hub_download
ids=[l.strip() for l in open(sys.argv[1]) if l.strip()][int(sys.argv[2]):int(sys.argv[3])]
def get(i):
    for f in ("adapter_config.json","adapter_model.safetensors"):
        hf_hub_download(i, f, local_dir=f"F:/lol/{i.split('/')[1]}")
    return i
with cf.ThreadPoolExecutor(8) as ex:
    for n,i in enumerate(ex.map(get, ids)): print(n, i, flush=True)
