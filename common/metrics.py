from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def mean_max_confidence(probs) -> float:
    probs = np.asarray(probs)
    return float(probs.max(axis=1).mean())


def consistency(preds_a, preds_b) -> float:
    """Fraction of examples whose predicted class is unchanged between two prediction arrays."""
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    return float((preds_a == preds_b).mean())


def per_class_accuracy(y_true, y_pred, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = y_true == c
        if mask.any():
            acc[c] = (y_pred[mask] == c).mean()
    return acc


def auroc(known_scores, unknown_scores) -> float:
    """AUROC for an unknownness score where larger = more novel.

    `known_scores` are the score's values on known (in-distribution) test examples,
    `unknown_scores` on out-of-distribution examples.
    """
    known_scores = np.asarray(known_scores)
    unknown_scores = np.asarray(unknown_scores)
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_score = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(y_true, y_score))


def rejection_threshold(known_val_scores, target_acceptance: float = 0.95) -> float:
    """Threshold tau such that `target_acceptance` fraction of known validation scores are <= tau.

    Accept x when u(x) <= tau. Calibrated on known validation data only, per the assignment's
    95th-percentile rejection protocol.
    """
    return float(np.percentile(np.asarray(known_val_scores), target_acceptance * 100.0))


def acceptance_rate(scores, threshold: float) -> float:
    """Fraction of examples with score <= threshold (i.e. accepted as known)."""
    return float((np.asarray(scores) <= threshold).mean())


def fpr_at_operating_point(unknown_scores, threshold: float) -> float:
    """FPR@95TPR under the assignment's convention: fraction of unknowns incorrectly accepted."""
    return acceptance_rate(unknown_scores, threshold)
