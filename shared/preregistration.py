from __future__ import annotations

"""Keeps the assignment's "state your expectation BEFORE interpreting the result" rule honest without
forcing the (slow) runs to wait for it:

  * `--blind` on the run scripts suppresses every result printout (per-epoch validation metrics, target
    accuracy, diagnostics, tables) while still writing all files, so a run can be launched now and its
    outputs left unread;
  * `assert_hypotheses_committed` is called at the top of each results-commit cell: it refuses to
    continue until the hypotheses file has no empty cells and is committed, so results cannot reach the
    repository before the expectations do.
"""

import os
import re
import subprocess


def _empty_cells(path: str):
    with open(path, encoding="utf-8") as f:
        return [line for line in f if re.search(r"\|\s*\|\s*$", line) and not set(line.strip()) <= set("|- ")]


def warn_if_unfilled(path: str) -> None:
    """Advisory reminder at launch time; never blocks a run."""
    if not os.path.exists(path):
        print(f"\n!!! WARNING: {path} not found -- write your expectations there before interpreting any result.\n")
    elif _empty_cells(path):
        print(f"\n!!! REMINDER: {path} still has empty cells. Fill it in and commit it BEFORE you read this run's results.\n")


def assert_hypotheses_committed(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    empty = _empty_cells(path)
    if empty:
        raise RuntimeError(f"{path} still has {len(empty)} empty cell(s). Write your hypotheses (your own words), commit them, then commit results.")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", path], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"{path} has uncommitted changes. Commit and push it first, then commit the results.")
    print(f"OK: {path} is filled in and committed.")


def blind_log(message: str) -> None:
    """Progress only: epoch counter and 'early stopping', no losses, F1 or accuracies."""
    if message.startswith("epoch"):
        print(message.split("|")[0].strip())
    elif message.startswith("early stopping"):
        print("early stopping")
