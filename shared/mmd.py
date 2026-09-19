from __future__ import annotations

from typing import Sequence

import torch

DEFAULT_BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)


def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    sq = (x * x).sum(dim=1, keepdim=True)
    return (sq + sq.t() - 2.0 * x @ x.t()).clamp_min(0.0)


def rbf_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth_multipliers: Sequence[float] = DEFAULT_BANDWIDTH_MULTIPLIERS,
) -> torch.Tensor:
    """Squared MMD between batches `x` and `y` using a sum of RBF kernels.

    Bandwidths are `bandwidth_multipliers` times the median pairwise squared distance of the
    combined batch, matching the assignment's kernel construction for both Task 2's DAN
    (source vs. target) and Task 3's DAN-DG (each pair of source domains) -- reused unchanged
    per the spec so the role of target access can be examined without also changing the
    discrepancy measure.
    """
    combined = torch.cat([x, y], dim=0)
    n = combined.size(0)
    sq_dists = _pairwise_sq_dists(combined)
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=combined.device)
    median_sq_dist = sq_dists[off_diag_mask].median().clamp_min(1e-8)

    kernel_sum = torch.zeros_like(sq_dists)
    for mult in bandwidth_multipliers:
        bandwidth = mult * median_sq_dist
        kernel_sum = kernel_sum + torch.exp(-sq_dists / (2.0 * bandwidth))

    nx = x.size(0)
    k_xx = kernel_sum[:nx, :nx]
    k_yy = kernel_sum[nx:, nx:]
    k_xy = kernel_sum[:nx, nx:]
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
