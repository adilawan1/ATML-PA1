from __future__ import annotations

"""Task 3 ERM baseline. This is the Task 2 Source-only checkpoint, reused unchanged --
do NOT retrain it here under a different configuration (per the assignment's shared-baseline
requirement)."""

import torch
import torch.nn as nn

from task2.models.backbone import build_resnet18_backbone


def load_erm_checkpoint(checkpoint_path: str = "task2/results/source_only/checkpoint.pt", device: str = "cuda") -> nn.Module:
    model = build_resnet18_backbone(num_classes=7).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    return model
