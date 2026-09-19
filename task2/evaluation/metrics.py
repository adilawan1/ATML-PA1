from __future__ import annotations

from typing import Dict

from common.metrics import accuracy, macro_f1


def evaluate_classifier(y_true, y_pred) -> Dict[str, float]:
    return {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred)}
