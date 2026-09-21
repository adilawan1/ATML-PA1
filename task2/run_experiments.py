from __future__ import annotations

"""Run all of Task 2 in one go (resumable): Source-only, DAN, DANN, CDAN, then the ONE controlled
design study you choose (`--study dan`: lambda_MMD in {0.1, 1, 10}; `--study dann`: max gradient-reversal
strength in {0.25, 0.5, 1}), then the final evaluation and curve figures.

Write your expected effect of stronger alignment on source performance, domain separability and target
recognition in `task2/hypotheses.md` and commit it BEFORE YOU READ THE RESULTS (run with --blind to launch
now and read later) -- and do not use the target results to change any setting (they are analysis-only).

Checkpoints go to `<ckpt-root>/task2/<run>/checkpoint.pt` (Drive), per-epoch logs and summaries to
`task2/results/<run>/`. A run whose `run_summary.json` exists is skipped unless `--force`.
"""

import argparse
import json
import os
import traceback
from typing import Dict

import torch

from shared.curves import plot_curves
from shared.pacs_data import PACSData
from shared.pacs_protocol import load_pacs_protocol
from shared.preregistration import blind_log, warn_if_unfilled
from shared.trainer import load_config, train_run
from task2.evaluate_final import evaluate_all
from task2.methods.cdan import CDAN
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.source_only import SourceOnly

CONFIG_DIR = "task2/configs"
RESULTS_DIR = "task2/results"

# run name -> (config file, method class, config overrides)
RUNS: Dict[str, tuple] = {
    "source_only": ("source_only.yaml", SourceOnly, {}),
    "dan": ("dan.yaml", DAN, {}),
    "dann": ("dann.yaml", DANN, {}),
    "cdan": ("cdan.yaml", CDAN, {}),
    "dan_lambda0.1": ("dan.yaml", DAN, {"loss": {"lambda_mmd": 0.1}}),
    "dan_lambda10": ("dan.yaml", DAN, {"loss": {"lambda_mmd": 10.0}}),
    "dann_grl0.25": ("dann.yaml", DANN, {"grl_schedule": {"max_alpha": 0.25}}),
    "dann_grl0.5": ("dann.yaml", DANN, {"grl_schedule": {"max_alpha": 0.5}}),
}
MAIN_RUNS = ["source_only", "dan", "dann", "cdan"]
STUDY_RUNS = {"dan": ["dan_lambda0.1", "dan", "dan_lambda10"], "dann": ["dann_grl0.25", "dann_grl0.5", "dann"]}


def main(args) -> None:
    warn_if_unfilled("task2/hypotheses.md")
    device = args.device
    blind = getattr(args, "blind", False)
    protocol = load_pacs_protocol(args.protocol)
    data = PACSData(protocol, args.pacs_root, cache_path=args.cache, include_target=True)

    plan = MAIN_RUNS + [r for r in STUDY_RUNS[args.study] if r not in MAIN_RUNS]
    if args.only:
        plan = [r for r in plan if r in args.only]
    checkpoint_for = lambda name: os.path.join(args.ckpt_root, "task2", name, "checkpoint.pt")  # noqa: E731

    failures = {}
    if not args.eval_only:
        for name in plan:
            out_dir = os.path.join(RESULTS_DIR, name)
            summary_path = os.path.join(out_dir, "run_summary.json")
            if os.path.exists(summary_path) and os.path.exists(checkpoint_for(name)) and not args.force:
                print(f"[skip] {name}: already trained")
                continue
            config_file, method_cls, overrides = RUNS[name]
            cfg = load_config(os.path.join(CONFIG_DIR, config_file), overrides)
            if args.max_epochs:  # debugging only; the real runs use the spec's 30
                cfg["train"]["max_epochs"] = args.max_epochs
            print(f"\n=== {name} ===")
            try:
                summary = train_run(method_cls, cfg, data, device, checkpoint_for(name), os.path.join(out_dir, "metrics.jsonl"), log=blind_log if blind else print)
                os.makedirs(out_dir, exist_ok=True)
                with open(summary_path, "w") as f:
                    json.dump({**summary, "overrides": overrides}, f, indent=2)
            except Exception:  # keep going: one failed run must not sink the rest of an unattended batch
                failures[name] = traceback.format_exc()
                print(failures[name])

    trained = [r for r in plan if os.path.exists(checkpoint_for(r)) and os.path.exists(os.path.join(RESULTS_DIR, r, "run_summary.json"))]
    if "source_only" in trained:
        evaluate_all(trained, data, checkpoint_for, RESULTS_DIR, device, MAIN_RUNS, verbose=not blind)
        os.makedirs(args.fig_dir, exist_ok=True)
        plot_curves({r: os.path.join(RESULTS_DIR, r, "metrics.jsonl") for r in MAIN_RUNS if r in trained}, os.path.join(args.fig_dir, "task2_curves.png"))
        study = [r for r in STUDY_RUNS[args.study] if r in trained]
        if len(study) > 1:
            plot_curves({r: os.path.join(RESULTS_DIR, r, "metrics.jsonl") for r in study}, os.path.join(args.fig_dir, f"task2_study_{args.study}_curves.png"))
    if failures:
        with open(os.path.join(RESULTS_DIR, "failures.json"), "w") as f:
            json.dump(failures, f, indent=2)
        print(f"\nFAILED RUNS: {list(failures)} (tracebacks in {RESULTS_DIR}/failures.json)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pacs-root", required=True)
    parser.add_argument("--ckpt-root", required=True, help="Where checkpoints go (Drive), e.g. <DRIVE_ROOT>/checkpoints")
    parser.add_argument("--study", required=True, choices=sorted(STUDY_RUNS), help="Which bounded controlled study to run")
    parser.add_argument("--protocol", default="shared/splits/pacs_sketch_seed6304.json")
    parser.add_argument("--cache", default="/content/pacs256_with_sketch.pt", help="Local-disk cache of the preprocessed images")
    parser.add_argument("--fig-dir", default="report/figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", nargs="*", help="Run only these run names")
    parser.add_argument("--max-epochs", type=int, default=None, help="debug only (spec: 30)")
    parser.add_argument("--force", action="store_true", help="Retrain even if a finished run exists")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--blind", action="store_true", help="hide all result printouts (write hypotheses first, read results after)")
    main(parser.parse_args())
