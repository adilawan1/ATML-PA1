from __future__ import annotations

from typing import Dict

import numpy as np

from common.metrics import acceptance_rate, auroc, rejection_threshold


def evaluate_score(
    known_val_scores: np.ndarray,
    known_test_scores: np.ndarray,
    near_scores: np.ndarray,
    far_scores: np.ndarray,
    target_acceptance: float = 0.95,
) -> Dict[str, float]:
    """One score's full Task 4 evaluation block: AUROC vs. near/far/all unknowns, plus the
    95th-percentile-on-known-val rejection protocol (accept x when u(x) <= tau).
    """
    all_unknown_scores = np.concatenate([near_scores, far_scores])
    tau = rejection_threshold(known_val_scores, target_acceptance)

    return {
        "auroc_near": auroc(known_test_scores, near_scores),
        "auroc_far": auroc(known_test_scores, far_scores),
        "auroc_all": auroc(known_test_scores, all_unknown_scores),
        "threshold_tau": tau,
        "known_test_acceptance_rate": acceptance_rate(known_test_scores, tau),
        "near_acceptance_rate_fpr95": acceptance_rate(near_scores, tau),
        "far_acceptance_rate_fpr95": acceptance_rate(far_scores, tau),
    }
