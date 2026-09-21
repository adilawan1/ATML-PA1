#!/usr/bin/env python
"""Pre-submission audit. Run from the repository root:  python tools/check_submission.py

Checks that every Required-Evidence item of the assignment (and every "split indices / configs / seeds /
attribution" deliverable) exists as a committed file with the expected content, that the hypotheses were
written and committed, and that the repository state is submittable (clean, pushed, public, no big files or
checkpoints). Prints PASS / MISSING / WARN per item and exits 1 if anything is MISSING.

  python tools/check_submission.py --write-map report/evidence_map.md   # also writes the evidence -> file map
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


# ------------------------------------------------------------------------------------------ validators
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def need_keys(*keys):
    def check(path):
        data = load_json(path)
        missing = [k for k in keys if k not in data]
        return (not missing, f"missing keys {missing}" if missing else "")
    return check


def csv_has_rows(*names, min_rows=0):
    def check(path):
        rows = csv_rows(path)
        first_col = {r[0] for r in rows[1:]}
        missing = [n for n in names if n not in first_col]
        if missing:
            return False, f"missing rows {missing}"
        return (len(rows) - 1 >= min_rows, f"needs >= {min_rows} data rows" if len(rows) - 1 < min_rows else "")
    return check


def check_subset(path):
    n = len(load_json(path)["indices"])
    return n == 500, f"{n} indices (expected 500)"


def check_cue_meta(path):
    meta = load_json(path)
    cells = meta["cells"]
    by_pair = {}
    for c in cells:
        by_pair.setdefault(c["pair"], set()).add(c["direction"])
    pairs = set(by_pair)
    kept = meta["totals"]["kept"]
    problems = []
    if kept < 200:
        problems.append(f"only {kept} conflicts kept (need >= 200)")
    if len(pairs) < 5:
        problems.append(f"only {len(pairs)} class pairs (need >= 5)")
    if any(len(d) < 2 for d in by_pair.values()):
        problems.append("not both directions for every pair")
    if "rejected_by_rule" not in meta["totals"]:
        problems.append("rejected count not recorded")
    return not problems, "; ".join(problems) or f"{kept} kept from {len(pairs)} pairs, {meta['totals']['rejected_by_rule']} rejected of {meta['totals']['candidates']}"


def check_cue_results(path):
    d = load_json(path)["predictors"]
    ok = all({"n_shape", "n_texture", "n_other", "shape_bias_pct", "coverage_pct"} <= set(v) for v in d.values()) and len(d) == 4
    return ok, "" if ok else "needs shape/texture/other counts, shape bias and coverage for all four predictors"


def check_table2(path):
    rows = load_json(path)
    have = {(r["method"], r["score"]) for r in rows}
    need = {("vanilla", "MLS"), ("gcsc", "MLS"), ("proser", "MLS"), ("proser", "placeholder")}
    miss = sorted(need - have)
    return not miss, f"missing rows {miss}" if miss else ""


def check_failures(path):
    d = load_json(path)
    ok = len(d.get("near", [])) >= 3 and len(d.get("far", [])) >= 3
    return ok, f"near={len(d.get('near', []))}, far={len(d.get('far', []))} (need >= 3 each)"


def check_task2_summary(path):
    d = load_json(path)
    miss = [r for r in ("source_only", "dan", "dann", "cdan") if r not in d]
    return not miss, f"missing runs {miss}" if miss else ""


def check_task3_summary(path):
    names = {r[0] for r in csv_rows(path)[1:]}
    miss = [r for r in ("erm", "dan_dg", "sam") if r not in names]
    return not miss, f"missing rows {miss}" if miss else ""


def hypotheses_ok(path):
    text = open(path, encoding="utf-8").read()
    empty = [l for l in text.splitlines() if re.search(r"\|\s*\|\s*$", l) and not set(l.strip()) <= set("|- ")]
    if empty:
        return False, f"{len(empty)} empty cell(s) -- write your hypotheses"
    dirty = run("git", "status", "--porcelain", "--", path)
    if dirty:
        return False, "uncommitted changes"
    log = run("git", "log", "--format=%H", "--", path)
    return bool(log), "" if log else "never committed"


# ------------------------------------------------------------------------------------------ the checklist
C = []  # (task, evidence item, [paths or globs], validator or None, severity)


def add(task, item, paths, validator=None, severity="required"):
    C.append((task, item, paths if isinstance(paths, list) else [paths], validator, severity))


# Task 1
add("Task 1", "Compact comparison: clean / grayscale / hue rotation / patch shuffle", "task1/results/compact_comparison.csv",
    csv_has_rows("resnet50", "vit_b_16", "clip_head", "clip_zeroshot"))
add("Task 1", "All clean-baseline, intervention, translation and stability numbers", "task1/results/task1_results.json",
    need_keys("conditions", "translation", "representation_stability", "tsne", "heads", "config"))
add("Task 1", "Translation curve (accuracy and consistency vs. displacement)", "report/figures/task1_translation_curve.png")
add("Task 1", "t-SNE of clean vs. transformed features (per backbone, joint fit)", "report/figures/task1_tsne.png")
add("Task 1", "Selected 500 evaluation image identifiers", "task1/data/eval_subset_seed6304.json", check_subset)
add("Task 1", "Stratified 80/20 train/val split indices", "task1/data/stl10_train_val_split_seed6304.json", need_keys("train", "val", "seed"))
add("Task 1", "Cue conflicts: rule, accepted/rejected counts, every kept item", "task1/results/cue_conflicts_meta.json", check_cue_meta)
add("Task 1", "Cue conflicts: shape/texture/other counts, shape bias, coverage", "task1/results/cue_conflicts.json", check_cue_results)
add("Task 1", "Cue-conflict agreements / disagreements / failures with predictions", "report/figures/task1_cue_conflict_examples.png")
add("Task 1", "Accepted and rejected cue-conflict contact sheets", ["report/figures/task1_cue_conflict_accepted.png", "report/figures/task1_cue_conflict_rejected.png"])
add("Task 1", "Cue-conflict representation visualization", "report/figures/task1_tsne_cue_conflict.png")
add("Task 1", "Hypotheses written and committed", "task1/hypotheses.md", hypotheses_ok)
# Task 2
add("Task 2", "PACS split protocol (seed 6304), shared with Task 3", "shared/splits/pacs_sketch_seed6304.json", need_keys("photo", "art_painting", "cartoon", "sketch", "seed"))
add("Task 2", "Main table: per-source val, mean, target acc/F1, change vs. Source-only, domain separability", ["task2/results/summary.json", "task2/results/summary.csv"], check_task2_summary)
add("Task 2", "Classification + alignment/domain-loss curves", "report/figures/task2_curves.png")
add("Task 2", "Per-class target changes, dominant confusions", "task2/results/class_analysis.json")
add("Task 2", "Confusion matrices (failure analysis)", "report/figures/task2_confusions.png")
add("Task 2", "Controlled alignment-strength study: table and plot", ["task2/results/study.csv", "report/figures/task2_study_*.png"], csv_has_rows(min_rows=3))
add("Task 2", "Per-epoch training logs of the four main runs", [f"task2/results/{r}/metrics.jsonl" for r in ("source_only", "dan", "dann", "cdan")])
add("Task 2", "Hypotheses written and committed", "task2/hypotheses.md", hypotheses_ok)
# Task 3
add("Task 3", "Main table: ERM / DAN-DG / SAM, per-source, mean, worst, Sketch acc/F1, change vs. ERM, separability, sharpness", "task3/results/summary.csv", check_task3_summary)
add("Task 3", "Sketch results incl. confusion matrices", "task3/results/sketch_results.json")
add("Task 3", "Source-side diagnostics (separability, sharpness) computed without Sketch", "task3/results/source_diagnostics.json")
add("Task 3", "Training curves incl. classification loss and MMD", "report/figures/task3_curves.png")
add("Task 3", "Per-class Sketch changes vs. ERM, dominant confusions", "task3/results/class_analysis.json")
add("Task 3", "Comparison with Task 2 (target-aware vs. target-free)", "task3/results/task2_vs_task3.csv")
add("Task 3", "Controlled study: table and plot", ["task3/results/study.csv", "report/figures/task3_study_*.png"], csv_has_rows(min_rows=3))
add("Task 3", "Confusion figure", "report/figures/task3_confusions.png")
add("Task 3", "Hypotheses written and committed", "task3/hypotheses.md", hypotheses_ok)
# Task 4
add("Task 4", "CIFAR-10 90/10 split indices", "task4/data/cifar10_split_seed6304.json", need_keys("train", "val", "seed"))
add("Task 4", "Table 1: MSP / MLS / Energy / Mahalanobis on Vanilla", "task4/results/table1_vanilla_posthoc_scores.json")
add("Task 4", "Table 2: Vanilla / GCSC / PROSER (MLS) + PROSER placeholder row", "task4/results/table2_model_comparison_mls.json", check_table2)
add("Task 4", "Score-distribution figure (MSP, MLS, Mahalanobis)", "report/figures/task4_score_distributions.png")
add("Task 4", ">= 3 near and >= 3 far incorrectly-accepted failures", "task4/results/failure_cases.json", check_failures)
add("Task 4", "Per-unknown-class acceptance and absorbing CIFAR-10 labels", "task4/results/unknown_class_breakdown.json")
add("Task 4", "PROSER hyperparameters and training log", ["task4/results/proser/run_config.json", "task4/results/proser/metrics.jsonl"])
add("Task 4", "Seeded Vanilla/GCSC training logs (their absence means they predate seeding)", ["task4/results/vanilla/metrics.jsonl", "task4/results/gcsc/metrics.jsonl"], severity="warn")
# Repository
add("Repo", "Top-level README with reproduction steps and attribution", "README.md", None)
add("Repo", "Environment specification", "requirements.txt")
add("Repo", "Recorded Colab environment (versions, GPU)", ["env/colab_environment.txt"], severity="warn")
add("Repo", "Colab notebook", "notebooks/colab_setup.ipynb")


# ------------------------------------------------------------------------------------------ helpers
def run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=60).stdout.strip()
    except Exception:
        return ""


def resolve(pattern):
    return [p for p in glob.glob(pattern) if os.path.isfile(p) and os.path.getsize(p) > 0]


def tracked(path):
    return bool(run("git", "ls-files", "--", path))


def audit():
    results = []
    for task, item, paths, validator, severity in C:
        missing = [p for p in paths if not resolve(p)]
        if missing:
            results.append((task, item, "MISSING" if severity == "required" else "WARN", "not found: " + ", ".join(missing)))
            continue
        uncommitted = [p for p in paths if not any(tracked(m) for m in resolve(p))]
        if uncommitted and task != "Repo":
            results.append((task, item, "MISSING" if severity == "required" else "WARN", "exists but not committed: " + ", ".join(uncommitted)))
            continue
        if validator:
            try:
                ok, msg = validator(resolve(paths[0])[0])
            except Exception as exc:  # malformed / unexpected content
                ok, msg = False, f"could not validate ({type(exc).__name__}: {exc})"
            if not ok:
                results.append((task, item, "MISSING" if severity == "required" else "WARN", msg))
                continue
            results.append((task, item, "PASS", msg))
        else:
            results.append((task, item, "PASS", ""))
    return results


def repo_state():
    out = []
    dirty = run("git", "status", "--porcelain")
    out.append(("Working tree clean (everything committed)", "PASS" if not dirty else "MISSING", "" if not dirty else f"{len(dirty.splitlines())} uncommitted path(s)"))
    run("git", "fetch", "origin", "main")
    head, remote = run("git", "rev-parse", "HEAD"), run("git", "rev-parse", "origin/main")
    out.append(("HEAD is pushed to origin/main", "PASS" if head and head == remote else "MISSING", "" if head == remote else "unpushed or diverged commits"))
    vis = run("gh", "repo", "view", "--json", "isPrivate", "--jq", ".isPrivate")
    out.append(("Repository is public", "PASS" if vis == "false" else ("WARN" if not vis else "MISSING"), "" if vis == "false" else ("could not check (gh unavailable)" if not vis else "repository is private")))
    files = run("git", "ls-files").splitlines()
    big = [f for f in files if os.path.isfile(f) and os.path.getsize(f) > 5_000_000]
    out.append(("No tracked file > 5 MB", "PASS" if not big else "MISSING", ", ".join(big)))
    banned = [f for f in files if re.search(r"\.(pt|pth|ckpt|parquet|npy)$", f) or f.startswith(("data/", "third_party/"))]
    out.append(("No checkpoints, datasets or third-party code tracked", "PASS" if not banned else "MISSING", ", ".join(banned[:5])))
    authors = set(run("git", "log", "--format=%an").splitlines())
    odd = [a for a in authors if len(a) < 3 or not re.search(r"[A-Za-z]{2}", a)]
    out.append(("Commit author names look like real names", "PASS" if not odd else "WARN", f"odd author name(s): {odd} -- set git config user.name for future commits" if odd else ""))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-map", metavar="PATH", help="write the evidence -> file map as markdown")
    parser.add_argument("--no-git", action="store_true", help="skip the git / GitHub state checks")
    args = parser.parse_args()

    if args.write_map:
        os.makedirs(os.path.dirname(args.write_map) or ".", exist_ok=True)
        with open(args.write_map, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Evidence map\n\nWhich committed file backs each Required-Evidence item (generated by `tools/check_submission.py`).\n\n")
            f.write("| Task | Evidence | File(s) |\n|---|---|---|\n")
            for task, item, paths, _, _ in C:
                f.write(f"| {task} | {item} | {', '.join(f'`{p}`' for p in paths)} |\n")
        print(f"wrote {args.write_map}")
        return 0

    results = audit()
    width = max(len(r[1]) for r in results)
    current = None
    for task, item, status, msg in results:
        if task != current:
            print(f"\n== {task}")
            current = task
        print(f"  [{status:7s}] {item:<{width}}  {msg}")
    problems = sum(r[2] == "MISSING" for r in results)
    warns = sum(r[2] == "WARN" for r in results)

    if not args.no_git:
        print("\n== Repository state")
        for item, status, msg in repo_state():
            print(f"  [{status:7s}] {item}  {msg}")
            problems += status == "MISSING"
            warns += status == "WARN"

    print(f"\n{sum(r[2] == 'PASS' for r in results)} passed, {problems} missing, {warns} warnings")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
