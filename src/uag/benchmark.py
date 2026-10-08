"""One-hour GPU benchmark: train one Stage-0-sized run and price it (compare GPUs before renting).

Trains the given base × task with the given training config for at most ``minutes`` of wall
clock (early stopping off, so the measured speed is sustained throughput including the
periodic validation passes), then extracts and verifies ΔW and evaluates the adapter on test
examples, exactly as the Stage-0 grid does. From the measured speeds it prints the time and
cost of one run trained to the full token cap (an upper bound: early stopping usually ends
runs sooner) and a Stage-0 training projection.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, load_base, load_lora, load_task, load_train

# Stage-0 shape: 4 bases x 10 tasks x 3 seeds; per run, direct evals on the task + 2 control
# tasks, each under 2 prompt templates.
STAGE0_RUNS = 120
DIRECT_EVALS_PER_RUN = 3 * 2


def run_benchmark(base_cfg: str, task_cfg: str, train_cfg: str, lora_cfg: str, minutes: float,
                  price_per_hour: float, eval_examples: int = 100, out_root: str = "artifacts") -> dict[str, Any]:
    from .evaluate import run_evaluation
    from .pipeline import Paths, train_grid
    from .provenance import hardware_info

    base, task, lora = load_base(base_cfg), load_task(task_cfg), load_lora(lora_cfg)
    train = load_train(train_cfg)
    cap = train.token_cap(task.task_id)
    bench_train = dataclasses.replace(train, early_stopping_patience=0, max_wall_minutes=minutes)
    gpu = (hardware_info().get("gpu") or "cpu").replace(" ", "_").replace("/", "_")
    exp = ExperimentConfig(name=f"benchmark_{gpu}_{int(time.time())}", bases=[base], tasks=[task], lora=lora,
                           train=bench_train, seeds=[0], require_pinned_revisions=False, artifacts_dir=out_root,
                           results_dir=str(Path(out_root) / "benchmark_results"))
    paths = Paths(exp)
    t0 = time.time()
    train_grid(exp)  # train + ΔW extraction + verification, as in Stage 0
    total_train = time.time() - t0
    run_dir = paths.run_dir(base.name, task.task_id, 0)
    m = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    post_seconds = max(total_train - m["wall_seconds"], 0.0)  # model reload + ΔW extraction + check

    s = run_evaluation(base, task, paths.raw, "bench_direct", adapter_dir=run_dir / "adapter", split="test",
                       max_examples=eval_examples, device=train.device, dtype=train.dtype)
    eval_per_example = s["eval_seconds"] / max(s["n"], 1)

    tps = m["tokens_seen"] / m["wall_seconds"]
    train_at_cap = cap / tps
    evals = DIRECT_EVALS_PER_RUN * task.n_test * eval_per_example
    per_run = train_at_cap + post_seconds + evals
    res = {
        "gpu": hardware_info().get("gpu"), "base": base.name, "task": task.dataset_key, "train_config": train_cfg,
        "benchmark_minutes": minutes, "stop_reason": m["stopping"]["reason"], "steps": m["steps"],
        "tokens_seen": m["tokens_seen"], "train_seconds": m["wall_seconds"], "tokens_per_second": tps,
        "seconds_per_step": m["wall_seconds"] / max(m["steps"], 1), "peak_vram_gb": (m["peak_vram_bytes"] or 0) / 1e9,
        "postprocess_seconds": post_seconds, "eval_seconds_per_example": eval_per_example,
        "token_cap": cap, "train_hours_at_cap": train_at_cap / 3600, "direct_eval_hours": evals / 3600,
        "hours_per_run": per_run / 3600, "price_per_hour": price_per_hour,
        "cost_per_run": per_run / 3600 * price_per_hour,
        "stage0_train_hours": STAGE0_RUNS * per_run / 3600,
        "stage0_train_cost": STAGE0_RUNS * per_run / 3600 * price_per_hour,
        "run_dir": str(run_dir),
    }
    out = Path(out_root) / "benchmark_results" / f"{exp.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    res["saved_to"] = str(out)
    return res


def render_benchmark(r: dict[str, Any]) -> str:
    done = "reached the token cap" if r["stop_reason"] == "token_cap" else \
        f"stopped by the {r['benchmark_minutes']:g}-minute limit; the rest is extrapolated"
    return "\n".join([
        f"GPU benchmark: {r['gpu']}  |  {r['base']} on {r['task']}  |  {r['train_config']}",
        f"  trained {r['tokens_seen']:,} tokens in {r['train_seconds'] / 60:.1f} min ({done})",
        f"  speed: {r['tokens_per_second']:,.0f} tokens/s, {r['seconds_per_step']:.2f} s/step "
        f"(validation passes included); peak VRAM {r['peak_vram_gb']:.1f} GB",
        f"  ΔW extraction + verification: {r['postprocess_seconds'] / 60:.1f} min; "
        f"evaluation: {r['eval_seconds_per_example']:.3f} s per test example",
        "",
        f"  One Stage-0 run at the full {r['token_cap']:,}-token cap (upper bound; early stopping is usually shorter):",
        f"    training {r['train_hours_at_cap']:.2f} h + direct evals {r['direct_eval_hours']:.2f} h "
        f"= {r['hours_per_run']:.2f} h  ->  ${r['cost_per_run']:.2f} at ${r['price_per_hour']:.2f}/h",
        f"  Stage-0 grid ({STAGE0_RUNS} runs, this model size for all; smaller bases are cheaper):",
        f"    {r['stage0_train_hours']:.0f} GPU-hours  ->  ${r['stage0_train_cost']:.0f} "
        f"(plus base / transfer evaluations)",
        f"  saved to {r['saved_to']}",
    ])
