from __future__ import annotations

import os
import re


def warn_if_unfilled(path: str) -> None:
    """The assignment wants each controlled study's expected effect stated BEFORE its results are
    interpreted. Advisory only: prints a loud warning when the hypotheses file is missing or still has
    empty table cells, and never blocks a run."""
    if not os.path.exists(path):
        print(f"\n!!! WARNING: {path} not found -- write your expectations there and commit it before launching.\n")
        return
    with open(path, encoding="utf-8") as f:
        empty = [line for line in f if re.search(r"\|\s*\|\s*$", line) and not set(line.strip()) <= set("|- ")]
    if empty:
        print(f"\n!!! WARNING: {path} still has {len(empty)} empty cell(s). State your expectations and commit "
              f"them BEFORE looking at any result from this run.\n")
