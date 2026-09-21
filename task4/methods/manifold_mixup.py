from __future__ import annotations

from typing import Optional, Tuple

import torch


def sample_cross_class_pairs(labels: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """For every example i, a random partner j with labels[j] != labels[i] (PROSER's data
    placeholders require y_i != y_j). Needs at least two classes in the batch."""
    different = (labels[:, None] != labels[None, :]).float()
    if not bool((different.sum(1) > 0).all()):
        raise ValueError("cross-class mixup needs at least two classes in the batch")
    return torch.multinomial(different, 1, generator=generator).squeeze(1)


def manifold_mixup(
    h: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0, beta: float = 2.0, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """h~_i = lambda_i * h_i + (1 - lambda_i) * h_j,  lambda_i ~ Beta(alpha, beta),  y_i != y_j
    (PROSER's data placeholders, applied to the layer2 feature map). Returns (mixed, partner_indices);
    the caller trains `mixed` toward the unknown/dummy class, never toward either original class."""
    partner = sample_cross_class_pairs(labels, generator)
    lam = torch.distributions.Beta(alpha, beta).sample((h.size(0),)).to(h.device)
    lam = lam.view(-1, *([1] * (h.dim() - 1)))
    return lam * h + (1.0 - lam) * h[partner], partner
