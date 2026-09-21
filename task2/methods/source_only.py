from __future__ import annotations

"""Source-only ERM (Task 2, Step 1): cross-entropy over the three labeled source domains with
domain-balanced batches. This is also the Task 3 ERM baseline -- Task 3 loads this checkpoint
rather than retraining it under a different configuration."""

from typing import Dict

import torch
import torch.nn.functional as F

from shared.trainer import Batch, Method


class SourceOnly(Method):
    def step(self, batch: Batch, optimizer, progress: float) -> Dict[str, torch.Tensor]:
        loss = F.cross_entropy(self.model(batch.source_x), batch.source_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {"cls_loss": loss.detach()}
