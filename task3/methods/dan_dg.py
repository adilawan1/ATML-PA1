from __future__ import annotations

"""DAN-DG -- pairwise MMD alignment across the three OBSERVED source domains only (Task 3,
Step 2). Never accesses Sketch. Uses the exact same MMD implementation/kernel construction as
Task 2's DAN (`shared.mmd.rbf_mmd2`), so the effect of target access can be examined without
also changing the discrepancy measure.

L_DAN-DG = L_ERM + (lambda_dg / 3) * sum_{e<e'} MMD^2(F(X_e), F(X_e'))  over {Photo, Art, Cartoon}.

TODO (Task 3, Day 4): extend `task2.methods.source_only`'s domain-balanced loop (reuse it
directly -- same batch construction, 8/domain, no target loader) to additionally compute the
three pairwise `rbf_mmd2` terms over `penultimate_features` for the current batch's three
domain slices, average them, and add `lambda_dg * mean_pairwise_mmd2` to the classification loss.
"""

from shared.mmd import rbf_mmd2  # noqa: F401
from task3.models.backbone import penultimate_features  # noqa: F401

LAMBDA_DG_MAIN = 1.0
BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)


def train_dan_dg(*args, **kwargs):
    raise NotImplementedError("Implement per the module docstring on Task 3's implementation day.")
