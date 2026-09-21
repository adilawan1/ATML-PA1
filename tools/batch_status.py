#!/usr/bin/env python
"""Did the overnight batch finish? Works from any fresh Colab session (only Google Drive has to be mounted) and
needs nothing from the repo clone -- so it does not disturb a batch that is still running:

    !curl -s https://raw.githubusercontent.com/adilawan1/ATML-PA1/main/tools/batch_status.py -o /tmp/batch_status.py
    !python /tmp/batch_status.py --drive-root /content/drive/MyDrive/atml_pa1

It inspects the durable artifacts the batch leaves on Drive (checkpoints, per-stage result backups, markers) and prints, per
stage, what exists, when it was written, and an overall verdict. Standard library only.
"""

import argparse
import csv
import json
import os
import subprocess
import time

TASK2_MAIN = ["source_only", "dan", "dann", "cdan"]
TASK2_STUDY = {"dan": ["dan_lambda0.1", "dan_lambda10"], "dann": ["dann_grl0.25", "dann_grl0.5"]}
TASK3_MAIN = ["dan_dg", "sam"]
TASK3_STUDY = {"dan_dg": ["dan_dg_lambda0.1", "dan_dg_lambda10"], "sam": ["sam_rho0.01", "sam_rho0.1"]}


class Report:
    def __init__(self):
        self.times, self.problems = [], []

    def line(self, ok, label, path=None, note=""):
        stamp = ""
        if path and os.path.exists(path):
            mtime = os.path.getmtime(path)
            self.times.append(mtime)
            stamp = time.strftime("%H:%M %b %d", time.localtime(mtime))
        if not ok:
            self.problems.append(label)
        print(f"  [{'ok' if ok else 'MISSING':7s}] {label:<58s} {stamp:>13s}  {note}")


def first(*paths):
    return next((p for p in paths if os.path.exists(p)), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive-root", required=True, help="e.g. /content/drive/MyDrive/atml_pa1")
    args = parser.parse_args()
    root = args.drive_root
    ckpt, bak = os.path.join(root, "checkpoints"), os.path.join(root, "results_backup")
    if not os.path.isdir(root):
        raise SystemExit(f"{root} not found -- mount Google Drive first")
    r = Report()

    print("\n== Task 2 (Source-only, DAN, DANN, CDAN + controlled study)")
    for run in TASK2_MAIN:
        c = os.path.join(ckpt, "task2", run, "checkpoint.pt")
        s = first(os.path.join(ckpt, "task2", run, "run_summary.json"), os.path.join(bak, "task2", run, "run_summary.json"))
        r.line(os.path.exists(c) and s is not None, f"{run}: checkpoint + run summary", c)
    study = next((k for k, runs in TASK2_STUDY.items() if any(os.path.exists(os.path.join(ckpt, "task2", x, "checkpoint.pt")) for x in runs)), None)
    r.line(study is not None and all(os.path.exists(os.path.join(ckpt, "task2", x, "checkpoint.pt")) for x in TASK2_STUDY.get(study, ["?"])),
           f"controlled study runs ({study or 'none found'})", os.path.join(ckpt, "task2"))
    for f in ("summary.json", "summary.csv", "class_analysis.json", "study.csv"):
        r.line(os.path.exists(os.path.join(bak, "task2", f)), f"backup task2/{f}", os.path.join(bak, "task2", f))

    print("\n== Task 3 (ERM, DAN-DG, SAM + controlled study, Sketch evaluation)")
    for run in TASK3_MAIN:
        c = os.path.join(ckpt, "task3", run, "checkpoint.pt")
        r.line(os.path.exists(c), f"{run}: checkpoint", c)
    study3 = next((k for k, runs in TASK3_STUDY.items() if any(os.path.exists(os.path.join(ckpt, "task3", x, "checkpoint.pt")) for x in runs)), None)
    r.line(study3 is not None and all(os.path.exists(os.path.join(ckpt, "task3", x, "checkpoint.pt")) for x in TASK3_STUDY.get(study3, ["?"])),
           f"controlled study runs ({study3 or 'none found'})", os.path.join(ckpt, "task3"))
    for f in ("source_diagnostics.json", "sketch_results.json", "summary.csv", "task2_vs_task3.csv", "study.csv"):
        r.line(os.path.exists(os.path.join(bak, "task3", f)), f"backup task3/{f}", os.path.join(bak, "task3", f))

    print("\n== Task 4 (Vanilla, GCSC, PROSER, final tables)")
    for m in ("vanilla", "gcsc", "proser"):
        c = os.path.join(ckpt, "task4", m, "checkpoint.pt")
        marker = c + ".seeded"
        r.line(os.path.exists(c), f"{m}: checkpoint", c, "seeded retrain done" if os.path.exists(marker) else "NOT marked seeded (unseeded, or retrain off)")
    t2 = os.path.join(bak, "task4", "table2_model_comparison_mls.json")
    have = set()
    if os.path.exists(t2):
        have = {(x["method"], x["score"]) for x in json.load(open(t2))}
    r.line(("proser", "MLS") in have and ("proser", "placeholder") in have, "backup task4 table 2 has both PROSER rows", t2)
    for f in ("unknown_class_breakdown.json", "failure_cases.json", "proser/run_config.json", "vanilla/metrics.jsonl", "gcsc/metrics.jsonl"):
        r.line(os.path.exists(os.path.join(bak, "task4", f)), f"backup task4/{f}", os.path.join(bak, "task4", f))

    print("\n== Task 1 (interventions; cue conflicts are a manual step)")
    for f in ("task1_results.json", "compact_comparison.csv"):
        r.line(os.path.exists(os.path.join(bak, "task1", f)), f"backup task1/{f}", os.path.join(bak, "task1", f))
    for f in ("eval_subset_seed6304.json", "stl10_train_val_split_seed6304.json"):
        r.line(os.path.exists(os.path.join(bak, f)), f"backup {f}", os.path.join(bak, f))

    print("\n== Figures backed up")
    for f in ("task1_translation_curve.png", "task1_tsne.png", "task2_curves.png", "task2_confusions.png", "task3_curves.png", "task3_confusions.png", "task4_score_distributions.png"):
        r.line(os.path.exists(os.path.join(bak, "figures", f)), f, os.path.join(bak, "figures", f))

    print("\n== Markers and code version")
    done = os.path.join(bak, "BATCH_FINISHED.txt")
    print(f"  BATCH_FINISHED.txt: {open(done).read().strip() if os.path.exists(done) else 'not present (only the newest batch cell writes it)'}")
    clone = "/content/ATML-PA1"
    if os.path.isdir(os.path.join(clone, ".git")):
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=clone, capture_output=True, text=True).stdout.strip()
        split = subprocess.run(["git", "status", "--porcelain", "--", "shared/splits/pacs_sketch_seed6304.json"], cwd=clone, capture_output=True, text=True).stdout.strip()
        print(f"  clone at commit {head}; PACS split {'is the committed file (unmodified)' if not split else 'DIFFERS from / is missing in git: ' + split}")
    else:
        print("  (no clone in this session)")

    if r.times:
        newest = max(r.times)
        age = (time.time() - newest) / 60
        print(f"\nnewest artifact: {time.strftime('%H:%M on %b %d', time.localtime(newest))} ({age:.0f} min ago)"
              + ("  -> written very recently: the batch may still be running" if age < 20 else ""))
    print("\nVERDICT:", "COMPLETE -- every stage's artifacts are on Drive" if not r.problems else f"INCOMPLETE -- {len(r.problems)} item(s) missing: " + "; ".join(r.problems[:6]) + (" ..." if len(r.problems) > 6 else ""))
    return 1 if r.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
