from __future__ import annotations

import torch.nn as nn


class LinearHead(nn.Module):
    """Standalone linear head, used where the classifier is trained separately from a frozen
    feature extractor (Task 1). Task 2/3 fine-tune ResNet-18's own `model.fc` instead.
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
