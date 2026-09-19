from __future__ import annotations

"""Checkpoint/hyperparameter selection for Task 3 -- mean macro-F1 across the three source
validation domains ONLY. No Sketch image may be loaded by this module or anything it calls;
that is enforced procedurally (this file never imports the target split), not just by convention.
"""

from typing import Dict

import torch.nn as nn
from torch.utils.data import DataLoader

from task2.methods.source_only import SOURCE_DOMAINS, mean_source_macro_f1

__all__ = ["SOURCE_DOMAINS", "mean_source_macro_f1", "select_by_mean_source_macro_f1"]


def select_by_mean_source_macro_f1(model: nn.Module, val_loaders: Dict[str, DataLoader], device: str) -> float:
    return mean_source_macro_f1(model, val_loaders, device)
