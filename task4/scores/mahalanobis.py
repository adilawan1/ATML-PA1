from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def fit_class_gaussians(
    train_features: np.ndarray, train_labels: np.ndarray, num_classes: int = 10, eps: float = 1e-6
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Class means and one shared diagonal covariance from UNAUGMENTED CIFAR-10 training
    features, per the assignment (diagonal Sigma with `eps` added to every entry).
    """
    means: Dict[int, np.ndarray] = {}
    for c in range(num_classes):
        means[c] = train_features[train_labels == c].mean(axis=0)

    centered = np.concatenate(
        [train_features[train_labels == c] - means[c] for c in range(num_classes)], axis=0
    )
    diag_var = centered.var(axis=0) + eps
    return means, diag_var


def mahalanobis_score(features: np.ndarray, means: Dict[int, np.ndarray], diag_var: np.ndarray) -> np.ndarray:
    """u_Mah(x) = min_c (f(x)-mu_c)^T Sigma^-1 (f(x)-mu_c), with diagonal Sigma."""
    inv_var = 1.0 / diag_var
    dists = np.stack(
        [((features - mu) ** 2 * inv_var).sum(axis=1) for mu in means.values()], axis=1
    )
    return dists.min(axis=1)
