"""Record inputs of the crossed side-split run that its evaluation fingerprints do not cover.

crossed_eval_sides.py names each evaluation by a fingerprint of its configuration (kind, task,
basis, rank, side, groups, adapter weight SHA-256s, evaluation settings, task-config SHA-256,
projection version). The fingerprint does not include the adapter configuration files (which set
the LoRA scaling), the contents of the test splits, or the resolved base-model revision. This
script records them and checks that none of these files was modified after the scoring run began
(the oldest summary in results/crossed_eval/raw_sides). Test splits are additionally verified
against the SHA-256 values in the committed dataset manifests every time they are loaded.

usage: python scripts/crossed_eval_provenance.py
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from uag.config import load_experiment
from uag.data import dataset_dir, manifest_path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "crossed_eval"
side = json.load(open(RES / "crossed_eval_sides.json"))
BASE = side["base"]
GROUP_SRC = {"shared_init0": ("dev_3080_crossed_shared", 0), "shared_init1": ("dev_3080_crossed_shared", 1),
             "indep_seed0": ("dev_a100_v2", 0)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mtime(p):
    return Path(p).stat().st_mtime


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


run_start = min(mtime(p) for p in (RES / "raw_sides").glob("*.summary.json"))
out = {"run_started_utc": iso(run_start), "adapter_config": {}, "test_split": {}, "base_model": {}}
ok = True
for g, (exp, seed) in GROUP_SRC.items():
    for t in side["tasks"]:
        p = ROOT / "artifacts" / exp / "runs" / f"{BASE}_{t}_seed{seed}_r16" / "adapter" / "adapter_config.json"
        cfg = json.loads(p.read_text())
        before = mtime(p) < run_start
        ok &= before
        out["adapter_config"][f"{g}/{t}"] = dict(sha256=sha(p), r=cfg["r"], lora_alpha=cfg["lora_alpha"],
                                                 scale=cfg["lora_alpha"] / cfg["r"], modified_utc=iso(mtime(p)),
                                                 unmodified_since_run_start=before)
exp = load_experiment(ROOT / "configs/experiments/dev_3080_crossed_shared.yaml", allow_unpinned=True)
tasks = {x.task_id: x for x in exp.tasks}
for t in side["tasks"]:
    mp, f = manifest_path(tasks[t], exp.path("data")), dataset_dir(tasks[t], exp.path("data")) / "test.jsonl"
    want = yaml.safe_load(mp.read_text())["test_sha256"]
    actual = sha(f)
    out["test_split"][t] = dict(manifest=str(mp.relative_to(ROOT)), manifest_test_sha256=want,
                                file=str(f.relative_to(ROOT)), file_sha256=actual, matches_manifest=actual == want)
    ok &= actual == want
hub = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
rev = (hub / "refs" / "main").read_text().strip()
snap = hub / "snapshots" / rev
weights = sorted(snap.glob("*.safetensors"))
out["base_model"] = dict(model_id="Qwen/Qwen2.5-0.5B-Instruct", resolved_revision=rev,
                         snapshots_in_cache=sorted(p.name for p in (hub / "snapshots").iterdir()),
                         weights={p.name: sha(p.resolve()) for p in weights},
                         weights_modified_utc={p.name: iso(mtime(p.resolve())) for p in weights},
                         unmodified_since_run_start=all(mtime(p.resolve()) < run_start for p in weights))
ok &= out["base_model"]["unmodified_since_run_start"] and len(out["base_model"]["snapshots_in_cache"]) == 1
out["all_checks_pass"] = bool(ok)
json.dump(out, open(RES / "crossed_eval_sides_provenance.json", "w"), indent=1)
print("all checks pass:", ok)
print("base revision:", rev, "snapshots:", out["base_model"]["snapshots_in_cache"])
print("scales:", sorted({v["scale"] for v in out["adapter_config"].values()}))
print({t: v["matches_manifest"] for t, v in out["test_split"].items()})
