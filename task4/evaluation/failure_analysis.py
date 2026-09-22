from __future__ import annotations

"""Incorrectly-accepted-unknown inspection (Task 4, Step 6): using the vanilla MLS threshold,
find unknown examples with score <= tau, i.e. accepted as a known class. Used by
`task4.evaluate_osr.build_failure_cases`, which selects >=3 near and >=3 far examples and
writes them to `task4/results/failure_cases.json`.
"""

from typing import Dict, List, Sequence

import numpy as np


def find_incorrect_acceptances(
    scores: np.ndarray, threshold: float, unknown_class_names: Sequence[str], predicted_classes: Sequence[str]
) -> List[Dict]:
    accepted_mask = scores <= threshold
    indices = np.flatnonzero(accepted_mask)
    return [
        {
            "index": int(i),
            "unknown_class": unknown_class_names[i],
            "predicted_known_class": predicted_classes[i],
            "score": float(scores[i]),
            "threshold": float(threshold),
        }
        for i in indices
    ]
