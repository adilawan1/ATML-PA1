from __future__ import annotations

import numpy as np


def msp_score(logits: np.ndarray) -> np.ndarray:
    """u_MSP(x) = 1 - max_k p_k(x). Larger = more novel (less confident)."""
    logits = logits - logits.max(axis=1, keepdims=True)  # numerically stable softmax
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return 1.0 - probs.max(axis=1)
