from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def energy_score(logits: np.ndarray) -> np.ndarray:
    """u_Energy(x) = -log sum_k exp(z_k(x)). Larger = more novel."""
    return -logsumexp(logits, axis=1)
