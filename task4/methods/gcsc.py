from __future__ import annotations

import torch.nn as nn
from torchvision import transforms

from task4.methods.vanilla import build_train_transform, train_vanilla


def train_gcsc(
    data_root: str,
    split_path: str = "task4/data/cifar10_split_seed6304.json",
    device: str = "cuda",
    checkpoint_path: str = "task4/results/gcsc/checkpoint.pt",
    metrics_path: str = "task4/results/gcsc/metrics.jsonl",
) -> nn.Module:
    """Exactly the Vanilla recipe (Task 4, Step 3) with one change: RandAugment(num_ops=2,
    magnitude=9) inserted after crop+flip, before ToTensor/Normalize. Evaluated with MLS to
    test whether an augmentation-induced closed-set-accuracy change comes with better rejection.
    """
    randaugment = transforms.RandAugment(num_ops=2, magnitude=9)
    train_transform = build_train_transform(extra=randaugment)
    return train_vanilla(
        data_root=data_root,
        split_path=split_path,
        device=device,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        train_transform=train_transform,
    )
