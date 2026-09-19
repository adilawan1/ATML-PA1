from __future__ import annotations

from typing import Tuple

import torch


def sample_cross_class_pairs(labels: torch.Tensor) -> torch.Tensor:
    """For each example in the batch, pick a random partner index with a DIFFERENT label
    (PROSER's data placeholders require y_i != y_j). Falls back to a random permutation and
    resolves any same-label collisions by local swapping.
    """
    n = labels.size(0)
    perm = torch.randperm(n, device=labels.device)
    same = labels[perm] == labels
    if same.any():
        idx = torch.nonzero(same, as_tuple=True)[0]
        shuffled = idx[torch.randperm(len(idx))]
        perm[idx] = perm[shuffled]
    return perm


def manifold_mixup(h: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0, beta: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mix layer2 features `h` between cross-class pairs: h~ = lambda*h_i + (1-lambda)*h_j,
    lambda ~ Beta(alpha, beta), y_i != y_j (PROSER's data placeholders, Task 4 Step 4).

    Returns (mixed_features, partner_indices) -- the caller trains `mixed_features` toward the
    dummy classifiers, never toward either original class.
    """
    partner_idx = sample_cross_class_pairs(labels)
    lam = torch.distributions.Beta(alpha, beta).sample((h.size(0),) + (1,) * (h.dim() - 1)).to(h.device)
    mixed = lam * h + (1.0 - lam) * h[partner_idx]
    return mixed, partner_idx
