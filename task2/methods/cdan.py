from __future__ import annotations

"""CDAN -- class-conditional adversarial alignment (Task 2, Step 4).

Identical to DANN (discriminator width/activation/dropout, GRL schedule, loss weight) except the
discriminator sees g(x) = vec(f (x) p), the outer product of the 512-d feature f and the classifier's
softmax p over the 7 classes (input dimension 512 * 7). No entropy conditioning, and neither f nor p is
detached, per the spec.
"""

import torch

from task2.methods.dann import DANN


def multilinear_map(features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """g(x) = vec(f (x) p): outer product of each row of `features` and `probs`, flattened."""
    return torch.bmm(features.unsqueeze(2), probs.unsqueeze(1)).flatten(1)


class CDAN(DANN):
    discriminator_input_dim = 512 * 7

    def discriminator_input(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return multilinear_map(features, torch.softmax(logits, dim=1))
