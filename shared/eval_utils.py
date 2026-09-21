from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from common.metrics import accuracy, macro_f1
from shared.pacs_data import SOURCE_DOMAINS, PACSData
from task2.models.backbone import penultimate_features


@torch.no_grad()
def predict(model: nn.Module, data: PACSData, rows: torch.Tensor, device: str, batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """(features, logits) on CPU for the given image rows, center-cropped, model in eval mode."""
    model.eval()
    feats, logits = [], []
    for start in range(0, len(rows), batch_size):
        x = data.batch_eval(rows[start : start + batch_size], device)
        f = penultimate_features(model, x)
        feats.append(f.cpu())
        logits.append(model.fc(f).cpu())
    return torch.cat(feats), torch.cat(logits)


def source_val_metrics(model: nn.Module, data: PACSData, device: str) -> Dict:
    """Accuracy / macro-F1 on each source validation split, plus their mean and worst-domain values."""
    per_domain = {}
    for domain in SOURCE_DOMAINS:
        _, logits = predict(model, data, data.rows(domain, "val"), device)
        preds = logits.argmax(1).numpy()
        labels = data.labels(domain, "val").numpy()
        per_domain[domain] = {"accuracy": accuracy(labels, preds), "macro_f1": macro_f1(labels, preds)}
    accs = [m["accuracy"] for m in per_domain.values()]
    f1s = [m["macro_f1"] for m in per_domain.values()]
    return {
        "per_domain": per_domain,
        "mean_accuracy": float(np.mean(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "worst_accuracy": float(np.min(accs)),
        "worst_macro_f1": float(np.min(f1s)),
    }
