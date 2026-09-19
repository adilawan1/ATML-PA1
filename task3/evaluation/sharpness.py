from __future__ import annotations

import copy

import torch
import torch.nn as nn


def local_sharpness(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, radius: float = 0.05) -> float:
    """Delta_sharp = L(theta + eps) - L(theta), eps = radius * grad / ||grad||_2.

    Standardized local diagnostic (Task 3, Step 4) -- evaluated on the SAME fixed validation
    batch (32 examples/source, seed 6304) for ERM, DAN-DG, and SAM, in eval mode. This is
    evidence about local stability under one specific perturbation, not a claim that the
    whole loss landscape is flatter.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    base_loss = criterion(model(images), labels)
    base_loss.backward()

    grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2).clamp_min(1e-12)

    perturbed_model = copy.deepcopy(model)
    with torch.no_grad():
        perturbed_params = [p for p, src in zip(perturbed_model.parameters(), model.parameters()) if src.grad is not None]
        for p_pert, g in zip(perturbed_params, grads):
            p_pert.add_(radius * g / grad_norm)

    with torch.no_grad():
        perturbed_loss = criterion(perturbed_model(images), labels)

    model.zero_grad()
    return float(perturbed_loss.item() - base_loss.item())
