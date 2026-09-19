from __future__ import annotations

"""Common evaluation (Task 4, Step 6): builds both required tables.

1. MSP/MLS/Energy/Mahalanobis on the frozen Vanilla model -- near/far/all AUROC and
   validation-calibrated rejection (`task4.evaluation.thresholds.evaluate_score`).
2. Vanilla/GCSC/PROSER compared with MLS as the common score, plus a PROSER row using its
   placeholder-based detection score. PROSER's CSA must use only the 10 known-class logits.

Also produces the required score-distribution/ROC figure (MSP, MLS, Mahalanobis) and pulls
>=3 near-unknown and >=3 far-unknown incorrectly-accepted examples via
`task4.evaluation.failure_analysis.find_incorrect_acceptances` under the Vanilla MLS threshold.

TODO (Task 4, Day 5, after `extract_outputs.py` has cached features/logits for every method):
load the cached tensors, apply each score in `task4/scores/`, call `evaluate_score` per
(model, score, unknown-group), and write `task4/results/summary.json` + figures under
`report/figures/` for the report.
"""

if __name__ == "__main__":
    raise NotImplementedError("Fill in per the module docstring once cached outputs exist.")
