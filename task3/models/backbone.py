from __future__ import annotations

# Task 3 reuses Task 2's backbone and BatchNorm policy unchanged (same architecture,
# initialization, and freeze-running-stats rule) -- re-exported here so task3/ has no
# cross-task import surprises for anyone reading only this folder.
from task2.models.backbone import (  # noqa: F401
    build_resnet18_backbone,
    freeze_batchnorm_stats,
    penultimate_features,
)
