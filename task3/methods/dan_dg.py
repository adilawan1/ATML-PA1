from __future__ import annotations

"""DAN-DG -- pairwise MMD alignment across the three OBSERVED source domains only (Task 3, Step 2).

    L = L_ERM + (lambda_DG / 3) * sum_{e<e'} MMD^2(F(X_e), F(X_e'))     e, e' in {Photo, Art, Cartoon}

Never touches Sketch. Uses the exact MMD implementation and 3-kernel construction of Task 2's DAN
(`shared.mmd.rbf_mmd2`; bandwidths 0.5/1/2 x the median pairwise squared distance, computed per domain
pair within the batch), so the role of target access can be examined without changing the discrepancy
measure. The batch is 8 images per domain, concatenated domain by domain, so each domain's features
are one equal-sized chunk.
"""

from itertools import combinations
from typing import Dict

import torch
import torch.nn.functional as F

from shared.mmd import rbf_mmd2
from shared.trainer import Batch, Method
from task2.models.backbone import penultimate_features


class DANDG(Method):
    uses_target = False

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)
        self.lam = cfg["loss"]["lambda_dg"]
        self.multipliers = cfg["loss"]["kernel_bandwidth_multipliers"]

    def step(self, batch: Batch, optimizer, progress: float) -> Dict[str, torch.Tensor]:
        features = penultimate_features(self.model, batch.source_x)
        cls_loss = F.cross_entropy(self.model.fc(features), batch.source_y)

        per_domain = features.chunk(batch.n_domains)
        pair_mmds = [rbf_mmd2(per_domain[i], per_domain[j], self.multipliers) for i, j in combinations(range(batch.n_domains), 2)]
        mmd = torch.stack(pair_mmds).mean()  # (1/3) * sum over the three unordered pairs
        loss = cls_loss + self.lam * mmd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {"cls_loss": cls_loss.detach(), "mmd": mmd.detach(), "total_loss": loss.detach()}
