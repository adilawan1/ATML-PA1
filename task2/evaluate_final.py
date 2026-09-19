from __future__ import annotations

"""Common evaluation + alignment diagnostic (Task 2, Step 5): run only after every checkpoint
and setting has been fixed. For each method, reports per-source-domain and target accuracy /
macro-F1, target accuracy change vs. Source-only, domain separability, and per-class target
accuracy shifts.

TODO (Task 2, Day 3, after all four methods are trained):
1. Load each method's checkpoint from its config's `output.checkpoint`.
2. Evaluate on each source val split (`task2.evaluation.metrics.evaluate_classifier`) and on
   the full target split from the shared protocol (labels used here for the first time).
3. Extract penultimate features for a balanced source-val/target sample and compute
   `task2.evaluation.domain_separability.domain_separability_score`.
4. Compute per-class target accuracy vs. Source-only with
   `task2.evaluation.class_analysis.per_class_accuracy` / `biggest_class_shifts`.
5. Write one combined table to `task2/results/summary.json` (or .csv) that the report reads from.
"""

if __name__ == "__main__":
    raise NotImplementedError("Fill in per the module docstring once all Task 2 checkpoints exist.")
