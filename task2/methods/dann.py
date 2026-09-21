from __future__ import annotations

"""DANN -- adversarial domain alignment via gradient reversal (Task 2, Step 3).

A discriminator (256-unit hidden layer, ReLU, dropout 0.5, 2-way output) sits on the 512-d feature
behind a gradient-reversal layer whose strength follows alpha(p) = 2/(1+exp(-10p)) - 1 (scaled by
`grl_schedule.max_alpha` for the controlled study; 1.0 in the main comparison). Only source examples
contribute to the class loss; source (label 0) and unlabeled target (label 1) both contribute to the
domain loss, with unit weight.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from shared.trainer import Batch, Method
from task2.models.backbone import penultimate_features
from task2.models.domain_discriminator import DomainDiscriminator, grl, grl_schedule


class DANN(Method):
    uses_target = True
    discriminator_input_dim = 512

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)
        d = cfg["discriminator"]
        self.disc = DomainDiscriminator(self.discriminator_input_dim, d["hidden_dim"], d["dropout"]).to(device)
        self.gamma = cfg["grl_schedule"]["gamma"]
        self.max_alpha = cfg["grl_schedule"].get("max_alpha", 1.0)
        self.domain_weight = cfg["loss"]["domain_loss_weight"]

    def extra_parameters(self):
        return self.disc.parameters()

    def set_train(self) -> None:
        self.disc.train()

    def discriminator_input(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return features

    def step(self, batch: Batch, optimizer, progress: float) -> Dict[str, torch.Tensor]:
        n_source = batch.source_x.size(0)
        features = penultimate_features(self.model, torch.cat([batch.source_x, batch.target_x]))
        logits = self.model.fc(features)
        cls_loss = F.cross_entropy(logits[:n_source], batch.source_y)

        alpha = self.max_alpha * grl_schedule(progress, self.gamma)
        domain_logits = self.disc(grl(self.discriminator_input(features, logits), alpha))
        domain_labels = torch.cat([torch.zeros(n_source), torch.ones(batch.target_x.size(0))]).long().to(features.device)
        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        loss = cls_loss + self.domain_weight * domain_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {
            "cls_loss": cls_loss.detach(),
            "domain_loss": domain_loss.detach(),
            "domain_acc": (domain_logits.argmax(1) == domain_labels).float().mean().detach(),
            "grl_alpha": alpha,
            "total_loss": loss.detach(),
        }
