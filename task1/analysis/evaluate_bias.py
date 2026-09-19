from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from common.metrics import accuracy, consistency, macro_f1, mean_max_confidence

__all__ = ["accuracy", "consistency", "macro_f1", "mean_max_confidence", "classify_cue_conflict_predictions", "shape_bias_and_coverage"]


def classify_cue_conflict_predictions(
    predictions: Sequence[int], shape_labels: Sequence[int], texture_labels: Sequence[int]
) -> np.ndarray:
    """Per Step 3: label each cue-conflict prediction as 0=shape/content, 1=texture/style,
    2=other (neither intended class).
    """
    predictions = np.asarray(predictions)
    shape_labels = np.asarray(shape_labels)
    texture_labels = np.asarray(texture_labels)

    out = np.full(len(predictions), 2, dtype=int)
    out[predictions == shape_labels] = 0
    out[predictions == texture_labels] = 1
    return out


def shape_bias_and_coverage(decision_codes: np.ndarray) -> Dict[str, float]:
    """Shape Bias(%) = N_shape / (N_shape + N_texture) * 100
    Coverage(%) = (N_shape + N_texture) / N_total * 100
    `decision_codes` from `classify_cue_conflict_predictions` (0=shape, 1=texture, 2=other).
    """
    n_total = len(decision_codes)
    n_shape = int((decision_codes == 0).sum())
    n_texture = int((decision_codes == 1).sum())
    denom = n_shape + n_texture

    return {
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_total - denom,
        "shape_bias_pct": 100.0 * n_shape / denom if denom > 0 else float("nan"),
        "coverage_pct": 100.0 * denom / n_total if n_total > 0 else float("nan"),
    }
