"""Command-line entry point: every stage is launched from an experiment config (spec §13)."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _quiet() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(argv: list[str] | None = None) -> int:
    _quiet()
    p = argparse.ArgumentParser(prog="uag", description="Universal Adaptation Geometry pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    def exp_cmd(name: str, help_: str):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--experiment", "-e", required=True, help="experiment config (YAML/JSON)")
        return sp

    sp = exp_cmd("validate", "validate an experiment config and print the resolved grid")
    sp.add_argument("--allow-unpinned", action="store_true",
                    help="preview a config whose hub revisions are not pinned yet (inspection only)")
    exp_cmd("make-data", "generate datasets + manifests for every task")
    sp = exp_cmd("train", "train the base × task × seed LoRA grid (idempotent)")
    sp.add_argument("--base", nargs="*")
    sp.add_argument("--task", nargs="*")
    sp.add_argument("--seed", nargs="*", type=int)
    sp = exp_cmd("eval", "evaluate raw bases, direct adapters or transferred updates")
    sp.add_argument("--stage", choices=["base", "direct", "transfer"], required=True)
    exp_cmd("declare-gate", "freeze the Stage-0 gate + splits before held-out evaluation")
    sp = exp_cmd("fit-maps", "fit pair maps on training tasks and predict held-out updates")
    sp.add_argument("--method", nargs="*")
    exp_cmd("analyze", "regenerate tables and the decision memo from raw results")
    exp_cmd("run", "run every stage in order")
    sp = exp_cmd("dry-run-report", "summarise the pre-rental checks (memory, ΔW check, lift, geometry, timing)")
    sp.add_argument("--min-lift", type=float, default=0.05)
    sp = exp_cmd("seed-compare", "is ΔW task signal or random-init noise? factor movement + seed comparison")
    sp.add_argument("--base", nargs="*")
    sp.add_argument("--task", nargs="*")

    sp = sub.add_parser("pin-revisions", help="resolve hub revisions to commit hashes and write them into base configs")
    sp.add_argument("configs", nargs="+", help="base config files (source: hub)")

    sp = sub.add_parser("spectral", help="print the spectral summary of one run")
    sp.add_argument("run_dir")

    args = p.parse_args(argv)
    if args.cmd == "pin-revisions":
        return pin_revisions(args.configs)
    if args.cmd == "spectral":
        from pathlib import Path

        summ = json.loads((Path(args.run_dir) / "spectral" / "spectral_summary.json").read_text())
        print(json.dumps({k: summ[k] for k in ("total_fro", "energy_by_layer", "energy_by_class")}, indent=1))
        return 0

    from . import pipeline
    from .config import ConfigError, load_experiment

    try:
        exp = load_experiment(args.experiment, allow_unpinned=getattr(args, "allow_unpinned", False))
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.cmd == "validate":
        n_runs = len(exp.bases) * len(exp.tasks) * len(exp.seeds)
        print(json.dumps({"name": exp.name, "bases": [b.name for b in exp.bases],
                          "tasks": [t.task_id for t in exp.tasks], "seeds": exp.seeds, "lora_runs": n_runs,
                          "splits": [s.split_id for s in exp.splits], "map_methods": exp.maps.methods,
                          "unpinned_bases": [b.name for b in exp.bases if not b.pinned]}, indent=1))
    elif args.cmd == "make-data":
        pipeline.prepare_data(exp)
    elif args.cmd == "train":
        pipeline.prepare_data(exp)
        pipeline.train_grid(exp, args.base, args.task, args.seed)
    elif args.cmd == "eval":
        {"base": pipeline.eval_bases, "direct": pipeline.eval_direct, "transfer": pipeline.eval_transfer}[args.stage](exp)
    elif args.cmd == "declare-gate":
        print(json.dumps(pipeline.declare_gate(exp), indent=1))
    elif args.cmd == "fit-maps":
        pipeline.fit_maps(exp, args.method)
    elif args.cmd == "analyze":
        from .analysis import analyze

        res = analyze(exp)
        print(f"wrote {res['decision_path']} ({res['n_cells']} transfer cells); "
              f"gate passed: {res['gate']['passed'] if res['gate'] else 'n/a'}")
    elif args.cmd == "dry-run-report":
        from .report import dry_run_report

        print(dry_run_report(exp, args.min_lift))
    elif args.cmd == "seed-compare":
        from .diagnostics import render_geometry_checks, save_seed_comparison

        print("\n".join(render_geometry_checks(exp, args.base, args.task)))
        print(f"\nwrote {save_seed_comparison(exp, args.base, args.task)}")
    elif args.cmd == "run":
        pipeline.run_all(exp)
    return 0


def pin_revisions(paths: list[str]) -> int:
    """Replace ``revision``/``tokenizer_revision`` with the current immutable commit hash."""
    import re
    from pathlib import Path

    import yaml
    from huggingface_hub import HfApi

    from .config import is_pinned_revision

    api = HfApi()
    for path in paths:
        text = Path(path).read_text()
        cfg = yaml.safe_load(text)
        if cfg.get("source", "hub") != "hub":
            continue
        if is_pinned_revision(cfg.get("revision")):
            print(f"{path}: already pinned ({cfg['revision']})")
            continue
        sha = api.model_info(cfg["model_id"], revision=cfg.get("revision") or "main").sha
        for key in ("revision", "tokenizer_revision"):
            text = re.sub(rf"^{key}:.*$", f"{key}: {sha}", text, flags=re.M)
        Path(path).write_text(text)
        print(f"{path}: pinned {cfg['model_id']} -> {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
