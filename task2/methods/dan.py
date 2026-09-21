from __future__ import annotations

"""DAN -- MMD alignment between source and target features (Task 2, Step 2).

L_DAN = L_cls + lambda_mmd * MMD^2(F(x_s), F(x_t)) on the 512-d feature before `model.fc`, using a sum
of three RBF kernels (bandwidths 0.5x / 1x / 2x the median pairwise squared distance of the combined
batch) from `shared.mmd`. Target images are unlabeled; the loss never sees which class they belong to.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from shared.mmd import rbf_mmd2
from shared.trainer import Batch, Method
from task2.models.backbone import penultimate_features


class DAN(Method):
    uses_target = True

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)
        self.lam = cfg["loss"]["lambda_mmd"]
        self.multipliers = cfg["loss"]["kernel_bandwidth_multipliers"]

    def step(self, batch: Batch, optimizer, progress: float) -> Dict[str, torch.Tensor]:
        n_source = batch.source_x.size(0)
        features = penultimate_features(self.model, torch.cat([batch.source_x, batch.target_x]))
        cls_loss = F.cross_entropy(self.model.fc(features[:n_source]), batch.source_y)
        mmd = rbf_mmd2(features[:n_source], features[n_source:], self.multipliers)
        loss = cls_loss + self.lam * mmd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {"cls_loss": cls_loss.detach(), "mmd": mmd.detach(), "total_loss": loss.detach()}
