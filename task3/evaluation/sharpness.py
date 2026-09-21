from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn

from shared.pacs_data import SOURCE_DOMAINS, PACSData

SEED = 6304


def fixed_validation_batch(data: PACSData, device: str, per_domain: int = 32, seed: int = SEED) -> Tuple[torch.Tensor, torch.Tensor]:
    """The one batch (32 validation images from each source, chosen with seed 6304) on which every
    model's sharpness proxy is measured, so the models are compared on identical inputs."""
    generator = torch.Generator().manual_seed(seed)
    rows, labels = [], []
    for domain in SOURCE_DOMAINS:
        pick = torch.randperm(len(data.rows(domain, "val")), generator=generator)[:per_domain]
        rows.append(data.rows(domain, "val")[pick])
        labels.append(data.labels(domain, "val")[pick])
    return data.batch_eval(torch.cat(rows), device), torch.cat(labels).to(device)


def local_sharpness(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, radius: float = 0.05) -> float:
    """Delta_sharp = L_val(theta + eps) - L_val(theta), eps = radius * grad / ||grad||_2 (one normalized
    gradient-ascent step), evaluated in eval mode. A standardized LOCAL diagnostic under one specific
    perturbation -- not a claim that the whole loss landscape is flatter."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    base_loss = criterion(model(images), labels)
    base_loss.backward()

    grads = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
    norm = torch.norm(torch.stack([g.norm(2) for g in grads if g is not None]), 2).clamp_min(1e-12)

    perturbed = copy.deepcopy(model)
    with torch.no_grad():
        for p, g in zip(perturbed.parameters(), grads):
            if g is not None:
                p.add_(radius * g / norm)
        perturbed_loss = criterion(perturbed(images), labels)
    model.zero_grad()
    return float(perturbed_loss.item() - base_loss.item())
