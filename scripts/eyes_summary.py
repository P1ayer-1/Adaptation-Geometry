"""Summary table for scripts/eyes_test.sh: does each kind of fixed A learn T1 / T2 / T3?"""
import json
from pathlib import Path

import yaml

from uag.budget import curve, curve_summary

VARIANTS = ["random16", "whitened16", "pca16", "random64", "trainable16"]
TASKS = ["T1_hidden_rule", "T2_nli", "T3_paraphrase"]
print(f"{'variant':<13}{'base':<18}{'task':<16}{'valid best':>11}{'test':>7}{'take-off (t50)':>16}{'stopped':>9}")
for v in VARIANTS:
    for b in ["dev_qwen2.5-0.5b", "dev_llama3.2-1b"]:
        for t in TASKS:
            rank = 64 if v == "random64" else 16
            d = Path(f"artifacts/eyes_{v}/runs/{b}_{t}_seed0_r{rank}")
            if not (d / "manifest.yaml").exists():
                print(f"{v:<13}{b:<18}{t:<16}{'(not run)':>11}")
                continue
            m = yaml.safe_load((d / "manifest.yaml").read_text())
            s = curve_summary(curve(d, "valid_primary"), lower_is_better=False)
            ev = Path(f"results/raw/eyes_{v}/direct__{b}__{t}__seed0__on_{t}__v1.summary.json")
            test = json.loads(ev.read_text())["primary"] if ev.exists() else None
            t50 = f"{s['t50'] // 1000}k tok" if s.get("t50") else "never"
            print(f"{v:<13}{b:<18}{t:<16}{m['selection']['value']:>11.3f}{(f'{test:.3f}' if test is not None else '–'):>7}"
                  f"{t50:>16}{m['stopping']['step']:>9}")
