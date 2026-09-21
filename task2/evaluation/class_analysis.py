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


def dominant_confusion(confusion: np.ndarray, cls: int) -> Tuple[int, float]:
    """For true class `cls`: (most frequent wrong predicted class, its share of that class's samples)."""
    row = confusion[cls].astype(float).copy()
    total = row.sum()
    row[cls] = -1
    wrong = int(np.argmax(row))
    return wrong, float(confusion[cls][wrong] / total) if total else 0.0
