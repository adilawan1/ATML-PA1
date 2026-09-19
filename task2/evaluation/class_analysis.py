from __future__ import annotations

from typing import Tuple

import numpy as np

from common.metrics import per_class_accuracy

__all__ = ["per_class_accuracy", "biggest_class_shifts"]


def biggest_class_shifts(
    acc_before: np.ndarray, acc_after: np.ndarray, k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (most-improved, most-degraded) class indices ranked by acc_after - acc_before.

    Used to find class-specific negative transfer hidden behind an aggregate target-accuracy gain.
    """
    delta = acc_after - acc_before
    order = np.argsort(delta)
    return order[-k:][::-1], order[:k]
