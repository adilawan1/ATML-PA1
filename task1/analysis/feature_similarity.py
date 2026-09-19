from __future__ import annotations

import torch


def cosine_stability(clean_features: torch.Tensor, transformed_features: torch.Tensor) -> float:
    """I_T = mean_i cosine_similarity(f(x_i), f(T(x_i))) (Step 6).

    Both tensors are (N, feature_dim), paired row-for-row (clean image i <-> its transformed
    counterpart T(x_i)).
    """
    cos = torch.nn.functional.cosine_similarity(clean_features, transformed_features, dim=1)
    return float(cos.mean().item())
