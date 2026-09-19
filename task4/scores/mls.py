from __future__ import annotations

import numpy as np


def mls_score(logits: np.ndarray) -> np.ndarray:
    """u_MLS(x) = -max_k z_k(x). Larger = more novel (lower max logit)."""
    return -logits.max(axis=1)
