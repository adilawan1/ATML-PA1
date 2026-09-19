from __future__ import annotations

"""DAN -- MMD alignment between source and target features (Task 2, Step 2).

L_DAN = L_cls + lambda_mmd * MMD^2(source_features, target_features), applied to the
512-d feature before `model.fc`. Uses `shared.mmd.rbf_mmd2` (same kernel construction reused
by Task 3's DAN-DG) with lambda_mmd = 1 for the main comparison.

TODO (Task 2, Day 3): adapt `task2.methods.source_only.train_source_only`'s loop to also
pull a target batch of 24 unlabeled Sketch images per step (via `shared.pacs.cycle_loader`
over the full target split in the protocol), compute
`shared.mmd.rbf_mmd2(penultimate_features(model, source_images), penultimate_features(model, target_images))`,
and add `lambda_mmd * mmd2` to the classification loss. Target labels must never be read here.
"""

from task2.models.backbone import penultimate_features  # noqa: F401  (used once the loop is filled in)
from shared.mmd import rbf_mmd2  # noqa: F401

LAMBDA_MMD_MAIN = 1.0
BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)


def train_dan(*args, **kwargs):
    raise NotImplementedError("Implement per the module docstring on Task 2's implementation day.")
