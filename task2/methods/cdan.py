from __future__ import annotations

"""CDAN -- class-conditional adversarial alignment (Task 2, Step 4).

Identical discriminator width/activation/dropout/GRL schedule/loss weight to DANN, but the
discriminator sees g(x) = vec(f (x) p) instead of f alone, where f = 512-d feature and
p = softmax(logits) over the 7 classes -- so the discriminator input dimension is 512 * 7.
No entropy conditioning; do not detach f or p (per spec).

TODO (Task 2, Day 3): reuse `task2.methods.dann`'s training loop structure, but build g(x)
via `multilinear_map` below before feeding `task2.models.domain_discriminator.DomainDiscriminator`.
"""

import torch

from task2.models.backbone import penultimate_features  # noqa: F401
from task2.models.domain_discriminator import DomainDiscriminator, grl, grl_schedule  # noqa: F401

NUM_CLASSES = 7
FEATURE_DIM = 512
DISCRIMINATOR_INPUT_DIM = FEATURE_DIM * NUM_CLASSES
DOMAIN_LOSS_WEIGHT = 1.0
GRL_GAMMA = 10.0


def multilinear_map(features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """g(x) = vec(f (x) p): outer product of each row of `features` and `probs`, flattened."""
    batch = features.size(0)
    outer = torch.bmm(features.unsqueeze(2), probs.unsqueeze(1))  # (batch, feature_dim, num_classes)
    return outer.view(batch, -1)


def train_cdan(*args, **kwargs):
    raise NotImplementedError("Implement per the module docstring on Task 2's implementation day.")
