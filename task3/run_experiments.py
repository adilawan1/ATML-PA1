from __future__ import annotations

"""Task 3 training + SOURCE-SIDE diagnostics (resumable). No Sketch image is loaded, decoded or cached
here: the data object is built with `include_target=False`. Sketch is first opened by
`task3/evaluate_sketch.py`, after everything below is fixed.

Runs DAN-DG (lambda_DG = 1) and SAM (rho = 0.05) -- the main comparison -- plus the ONE controlled study
you choose (`--study dan_dg`: lambda_DG in {0.1, 1, 10}; `--study sam`: rho in {0.01, 0.05, 0.1}). ERM is
the Task 2 Source-only checkpoint, loaded unchanged. Then, for ERM and every trained model: per-source
validation metrics, source-domain separability (3-way Photo/Art/Cartoon logistic regression on balanced
frozen features, 70/30 split, seed 6304, chance = 33.3%), and the shared-batch local sharpness proxy.

Write your expected effects in `task3/hypotheses.md` and commit it BEFORE YOU READ THE RESULTS (--blind lets you
launch now). Task 2's Sketch
results must not be used to change any Task 3 setting.
"""

import argparse
import json
import os
import traceback

import numpy as np
import torch

from shared.curves import plot_curves
from shared.eval_utils import predict, source_val_metrics
from shared.pacs_data import SOURCE_DOMAINS, PACSData
from shared.pacs_protocol import load_pacs_protocol
from shared.preregistration import blind_log, warn_if_unfilled
from shared.trainer import load_config, train_run
from task2.evaluate_final import load_model
from task3.evaluation.sharpness import fixed_validation_batch, local_sharpness
from task3.evaluation.source_domain_separability import source_domain_separability_score
from task3.methods.dan_dg import DANDG
from task3.methods.sam import SAM

CONFIG_DIR = "task3/configs"
RESULTS_DIR = "task3/results"
SEED = 6304

RUNS = {
    "dan_dg": ("dan_dg.yaml", DANDG, {}),
    "sam": ("sam.yaml", SAM, {}),
    "dan_dg_lambda0.1": ("dan_dg.yaml", DANDG, {"loss": {"lambda_dg": 0.1}}),
    "dan_dg_lambda10": ("dan_dg.yaml", DANDG, {"loss": {"lambda_dg": 10.0}}),
    "sam_rho0.01": ("sam.yaml", SAM, {"sam": {"rho": 0.01}}),
    "sam_rho0.1": ("sam.yaml", SAM, {"sam": {"rho": 0.1}}),
}
MAIN_RUNS = ["dan_dg", "sam"]
STUDY_RUNS = {"dan_dg": ["dan_dg_lambda0.1", "dan_dg", "dan_dg_lambda10"], "sam": ["sam_rho0.01", "sam", "sam_rho0.1"]}


def checkpoint_for(ckpt_root: str, name: str) -> str:
    if name == "erm":  # the Task 2 Source-only model, unchanged
        return os.path.join(ckpt_root, "task2", "source_only", "checkpoint.pt")
    return os.path.join(ckpt_root, "task3", name, "checkpoint.pt")


def source_diagnostics(model, data: PACSData, device: str, sharp_batch) -> dict:
    per_domain_feats = [predict(model, data, data.rows(d, "val"), device)[0].numpy() for d in SOURCE_DOMAINS]
    n = min(len(f) for f in per_domain_feats)  # balanced: same number of validation images per source
    rng = np.random.RandomState(SEED)
    balanced = [f[rng.choice(len(f), size=n, replace=False)] for f in per_domain_feats]
    return {
        "source_val": source_val_metrics(model, data, device),
        "source_domain_separability": source_domain_separability_score(*balanced),
        "sharpness": local_sharpness(model, *sharp_batch, radius=0.05),
    }


def main(args) -> None:
    warn_if_unfilled("task3/hypotheses.md")
    device = args.device
    blind = getattr(args, "blind", False)
    protocol = load_pacs_protocol(args.protocol)
    data = PACSData(protocol, args.pacs_root, cache_path=args.cache, include_target=False)
    assert not data.include_target
    erm_ckpt = checkpoint_for(args.ckpt_root, "erm")
    if not os.path.exists(erm_ckpt):
        raise FileNotFoundError(f"Task 3 needs the Task 2 Source-only checkpoint at {erm_ckpt} (run Task 2 first; do not retrain it here)")

    plan = MAIN_RUNS + [r for r in STUDY_RUNS[args.study] if r not in MAIN_RUNS]
    if args.only:
        plan = [r for r in plan if r in args.only]

    failures = {}
    if not args.diagnostics_only:
        for name in plan:
            out_dir = os.path.join(RESULTS_DIR, name)
            summary_path = os.path.join(out_dir, "run_summary.json")
            if os.path.exists(summary_path) and os.path.exists(checkpoint_for(args.ckpt_root, name)) and not args.force:
                print(f"[skip] {name}: already trained")
                continue
            config_file, method_cls, overrides = RUNS[name]
            cfg = load_config(os.path.join(CONFIG_DIR, config_file), overrides)
            if args.max_epochs:  # debugging only; the real runs use the spec's 30
                cfg["train"]["max_epochs"] = args.max_epochs
            print(f"\n=== {name} ===")
            try:
                summary = train_run(method_cls, cfg, data, device, checkpoint_for(args.ckpt_root, name), os.path.join(out_dir, "metrics.jsonl"), log=blind_log if blind else print)
                with open(summary_path, "w") as f:
                    json.dump({**summary, "overrides": overrides}, f, indent=2)
            except Exception:
                failures[name] = traceback.format_exc()
                print(failures[name])

    trained = [r for r in plan if os.path.exists(os.path.join(RESULTS_DIR, r, "run_summary.json")) and os.path.exists(checkpoint_for(args.ckpt_root, r))]
    sharp_batch = fixed_validation_batch(data, device)
    diagnostics = {}
    for name in ["erm"] + trained:
        diagnostics[name] = source_diagnostics(load_model(checkpoint_for(args.ckpt_root, name), device), data, device, sharp_batch)
        d = diagnostics[name]
        if not blind:
            print(f"{name:18s} src F1 mean/worst {d['source_val']['mean_macro_f1']:.4f}/{d['source_val']['worst_macro_f1']:.4f} | "
                  f"source-domain separability {d['source_domain_separability']:.4f} | sharpness {d['sharpness']:.5f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "source_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    curve_runs = {r: os.path.join(RESULTS_DIR, r, "metrics.jsonl") for r in MAIN_RUNS if r in trained}
    erm_log = "task2/results/source_only/metrics.jsonl"
    if os.path.exists(erm_log):
        curve_runs = {"erm (task 2 source-only)": erm_log, **curve_runs}
    if curve_runs:
        os.makedirs(args.fig_dir, exist_ok=True)
        plot_curves(curve_runs, os.path.join(args.fig_dir, "task3_curves.png"))
    if failures:
        with open(os.path.join(RESULTS_DIR, "failures.json"), "w") as f:
            json.dump(failures, f, indent=2)
        print(f"\nFAILED RUNS: {list(failures)} (tracebacks in {RESULTS_DIR}/failures.json)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pacs-root", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--study", required=True, choices=sorted(STUDY_RUNS), help="Which bounded controlled study to run")
    parser.add_argument("--protocol", default="shared/splits/pacs_sketch_seed6304.json")
    parser.add_argument("--cache", default="/content/pacs256_source_only.pt", help="Cache of preprocessed SOURCE images (no Sketch)")
    parser.add_argument("--fig-dir", default="report/figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-epochs", type=int, default=None, help="debug only (spec: 30)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--blind", action="store_true", help="hide all result printouts")
    main(parser.parse_args())
