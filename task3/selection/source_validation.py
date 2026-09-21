from __future__ import annotations

"""Checkpoint / setting selection for Task 3: mean macro-F1 across the three SOURCE validation
domains, and nothing else. `shared.trainer.train_run` applies it every epoch. This module imports no
target-domain code and takes a `PACSData` that may have been built with `include_target=False`, which
is how the Task 3 training script never opens a Sketch file."""

from shared.eval_utils import source_val_metrics
from shared.pacs_data import PACSData


def mean_source_macro_f1(model, data: PACSData, device: str) -> float:
    return source_val_metrics(model, data, device)["mean_macro_f1"]
