"""Head-start test summary: zero-default transfer, shared-core head start vs LoRA from scratch."""
import json
from collections import defaultdict
from pathlib import Path

root = Path("results")
print("== Zero-default shared adapter: held-out transfer (no target training)")
print((root / "shared_adapter_zero_d64_gpu" / "shared_adapter.txt").read_text())
rows = json.loads((root / "shared_adapter_zero_d64_gpu" / "headstart.json").read_text())
lora = {}
for n in (32, 128, 512):
    for f in (root / "raw" / f"headstart_lora_n{n}").glob("direct__*.summary.json"):
        s = json.loads(f.read_text())
        lora[(s["task_id"], s["target_base"], n)] = s["primary"]
print("== Head start: test score after training on the TARGET from N examples")
by = defaultdict(dict)
for r in rows:
    by[(r["task"], r["target"], r["n"])][r["init"]] = r
inits = ["zero (from scratch)", "mean training core"]
print(f"{'task':<15}{'target':<18}{'N':>5}  {'zero':>8}{'mean':>8}{'transfer':>10}{'LoRA':>8}   (valid at step 0: zero/mean/transfer)")
for (t, b, n), d in sorted(by.items()):
    tr = next((v for k, v in d.items() if k.startswith("transferred")), None)
    cells = [d[i]["test_score"] if i in d else float("nan") for i in inits]
    lo = lora.get((t, b, n))
    starts = "/".join(f"{x['start_valid']:.2f}" for x in [d.get(inits[0]), d.get(inits[1]), tr] if x)
    print(f"{t:<15}{b:<18}{n:>5}  {cells[0]:>8.3f}{cells[1]:>8.3f}{(tr['test_score'] if tr else float('nan')):>10.3f}"
          f"{(lo if lo is not None else float('nan')):>8.3f}   ({starts})")
print("\n== Head start: gold-answer NLL on test (lower is better)")
for (t, b, n), d in sorted(by.items()):
    tr = next((v for k, v in d.items() if k.startswith("transferred")), None)
    print(f"{t:<15}{b:<18}{n:>5}  zero {d[inits[0]]['test_nll']:.3f}  mean {d[inits[1]]['test_nll']:.3f}  "
          f"transfer {tr['test_nll'] if tr else float('nan'):.3f}")
