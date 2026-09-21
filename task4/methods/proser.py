from __future__ import annotations

"""PROSER -- classifier and data placeholders (Task 4, Step 4; Zhou et al., CVPR 2021).

Initialized from the selected Vanilla checkpoint, with 5 randomly initialized dummy classifiers (linear
heads on the 512-d penultimate feature) appended. The (K+1)-th logit is the MAX over the dummy heads:

    f^(x) = [ W^T phi(x),  max_k w_k^T phi(x) ]                                   (paper, multiple dummies)

Each mini-batch is split in two equal halves:
  * first half -- classifier placeholders (Eq. 5):  l1 = CE(f^(x), y) + beta * CE(f^(x)\\y, K+1), where
    f^(x)\\y removes the true class's logit, so a dummy must become the runner-up (beta = 1);
  * second half -- data placeholders (Eq. 7): layer2 feature maps of two examples from DIFFERENT classes are
    mixed, h~ = lam*h_i + (1-lam)*h_j with lam ~ Beta(2, 2), passed through the rest of the network, and
    trained toward class K+1:  l2 = CE(f^(h~), K+1);
  * objective  L = l1 + gamma * l2  (gamma = 0.1).
No CIFAR-100 image participates; checkpoint selection uses CIFAR-10 validation accuracy computed from the
ten KNOWN-class logits only (so classification and rejection stay distinct).

Detection score (the authors' reference implementation, `valdummy`): softmax with temperature 1024 over
[known logits, max-dummy logit]; unknownness = P(dummy) - max_k P(known_k). With T = 1024 this is
essentially the logit gap (max-dummy - max-known), i.e. the paper's "dummy vs. known + bias" rule; the
bias is fixed by the 95% known-acceptance threshold on CIFAR-10 validation scores.

Reference: github.com/zhoudw-zdw/CVPR21-Proser (WideResNet, K=6). Differences here, all per the assignment:
CIFAR ResNet-18, 10 known classes, 5 dummy heads combined by max (the paper's formulation; the released code
matches it only for a single dummy), SGD lr 1e-3 / 50 epochs / cosine, first half = classifier placeholders.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.logging import MetricLogger, get_logger
from common.metrics import accuracy
from common.seed import set_seed
from task4.methods.manifold_mixup import manifold_mixup
from task4.methods.vanilla import build_cifar10_loaders
from task4.models.resnet_cifar import forward_from_layer3, penultimate_features, resnet18_cifar, split_forward_before_layer3

NUM_KNOWN = 10
NUM_DUMMY = 5
BETA_CLASSIFIER_PLACEHOLDER = 1.0
GAMMA_DATA_PLACEHOLDER = 0.1
MIXUP_ALPHA = 2.0
DETECTION_TEMPERATURE = 1024.0


class PROSERNet(nn.Module):
    def __init__(self, base: nn.Module, num_dummy: int = NUM_DUMMY):
        super().__init__()
        self.base = base
        self.dummy = nn.Linear(base.fc.in_features, num_dummy)  # dummy classifiers on the penultimate feature

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = penultimate_features(self.base, x)
        return self.base.fc(feat), self.dummy(feat)

    def to_layer2(self, x: torch.Tensor) -> torch.Tensor:
        return split_forward_before_layer3(self.base, x)

    def from_layer2(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, known = forward_from_layer3(self.base, h)
        return known, self.dummy(feat)


def extend_logits(known: torch.Tensor, dummy: torch.Tensor) -> torch.Tensor:
    """(K+1)-way logits: known logits plus the strongest dummy response."""
    return torch.cat([known, dummy.max(dim=1, keepdim=True).values], dim=1)


def proser_losses(
    model: PROSERNet, x: torch.Tensor, y: torch.Tensor, beta: float, gamma: float, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    half = x.size(0) // 2
    unknown = NUM_KNOWN  # index of the (K+1)-th class

    # first half: classifier placeholders (Eq. 5)
    xa, ya = x[:half], y[:half]
    logits = extend_logits(*model(xa))
    ce = F.cross_entropy(logits, ya)
    without_true = logits.scatter(1, ya.unsqueeze(1), -1e9)  # f^(x)\y: drop the ground-truth class
    placeholder = F.cross_entropy(without_true, torch.full_like(ya, unknown))

    # second half: data placeholders (Eq. 7) -- manifold mixup of different-class layer2 features
    xb, yb = x[half:], y[half:]
    mixed, _ = manifold_mixup(model.to_layer2(xb), yb, alpha=MIXUP_ALPHA, beta=MIXUP_ALPHA, generator=generator)
    data_placeholder = F.cross_entropy(extend_logits(*model.from_layer2(mixed)), torch.full_like(yb, unknown))

    loss = ce + beta * placeholder + gamma * data_placeholder
    return loss, {"ce": ce.detach(), "classifier_placeholder": placeholder.detach(), "data_placeholder": data_placeholder.detach()}


@torch.no_grad()
def known_accuracy(model: PROSERNet, loader: DataLoader, device: str) -> float:
    model.eval()
    preds, labels = [], []
    for images, target in loader:
        preds.extend(model(images.to(device))[0].argmax(1).cpu().tolist())  # known-class logits only
        labels.extend(target.tolist())
    return accuracy(labels, preds)


def proser_placeholder_score(known_logits: np.ndarray, dummy_logits: np.ndarray, temperature: float = DETECTION_TEMPERATURE) -> np.ndarray:
    """Unknownness = P(dummy) - max_k P(known_k) under softmax(temperature-scaled [known, max dummy])."""
    extended = np.concatenate([known_logits, dummy_logits.max(axis=1, keepdims=True)], axis=1) / temperature
    extended = extended - extended.max(axis=1, keepdims=True)
    probs = np.exp(extended)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs[:, -1] - probs[:, :-1].max(axis=1)


def load_proser(checkpoint_path: str, device: str) -> PROSERNet:
    model = PROSERNet(resnet18_cifar(NUM_KNOWN)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    return model.eval()


def train_proser(
    data_root: str,
    vanilla_checkpoint: str,
    split_path: str = "task4/data/cifar10_split_seed6304.json",
    device: str = "cuda",
    checkpoint_path: str = "task4/results/proser/checkpoint.pt",
    metrics_path: str = "task4/results/proser/metrics.jsonl",
    epochs: int = 50,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 128,
    seed: int = 6304,
    max_steps_per_epoch: Optional[int] = None,  # debugging only
) -> PROSERNet:
    set_seed(seed)
    import json

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(os.path.join(os.path.dirname(metrics_path), "run_config.json"), "w") as f:
        json.dump(
            {"seed": seed, "epochs": epochs, "lr": lr, "momentum": momentum, "weight_decay": weight_decay, "batch_size": batch_size,
             "beta_classifier_placeholder": BETA_CLASSIFIER_PLACEHOLDER, "gamma_data_placeholder": GAMMA_DATA_PLACEHOLDER,
             "num_dummy": NUM_DUMMY, "mixup_beta_alpha": MIXUP_ALPHA, "detection_temperature": DETECTION_TEMPERATURE,
             "initialized_from": vanilla_checkpoint, "cosine_schedule": True},
            f, indent=2,
        )
    logger = get_logger("task4_proser")
    metric_logger = MetricLogger(metrics_path)
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    base = resnet18_cifar(NUM_KNOWN)
    base.load_state_dict(torch.load(vanilla_checkpoint, map_location="cpu")["model_state"])
    model = PROSERNet(base).to(device)

    loaders = build_cifar10_loaders(data_root, split_path, batch_size)  # standard crop + flip augmentation
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = -1.0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    for epoch in range(epochs):
        model.train()
        sums: Dict[str, float] = {}
        steps = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            loss, parts = proser_losses(model, images, labels, BETA_CLASSIFIER_PLACEHOLDER, GAMMA_DATA_PLACEHOLDER)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in {**parts, "loss": loss.detach()}.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            steps += 1
            if max_steps_per_epoch and steps >= max_steps_per_epoch:
                break
        scheduler.step()

        val_acc = known_accuracy(model, loaders["val"], device)
        row = {"epoch": epoch, **{k: v / steps for k, v in sums.items()}, "val_accuracy": val_acc}
        metric_logger.log(**row)
        logger.info(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_accuracy": val_acc}, checkpoint_path)

    return load_proser(checkpoint_path, device)
