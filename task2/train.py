from __future__ import annotations

"""Train (and evaluate) a single named Task 2 run, e.g. to debug or rerun one method:

    python -m task2.train --run dann --study dann --pacs-root /content/pacs --ckpt-root <DRIVE>/checkpoints

Same as `task2.run_experiments --only <run>`; the full batch is `task2/run_experiments.py`.
"""

import argparse

import torch

from task2.run_experiments import RUNS, STUDY_RUNS, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, choices=sorted(RUNS))
    parser.add_argument("--study", default="dan", choices=sorted(STUDY_RUNS))
    parser.add_argument("--pacs-root", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--protocol", default="shared/splits/pacs_sketch_seed6304.json")
    parser.add_argument("--cache", default="/content/pacs256_with_sketch.pt")
    parser.add_argument("--fig-dir", default="report/figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None, help="debug only (spec: 30)")
    ns = parser.parse_args()
    ns.only, ns.eval_only = [ns.run], False
    main(ns)
