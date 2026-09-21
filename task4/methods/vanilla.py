from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from common.logging import MetricLogger, get_logger
from common.metrics import accuracy
from common.seed import set_seed
from task4.data.cifar10 import load_cifar10_split
from task4.models.resnet_cifar import resnet18_cifar

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_train_transform(extra: Optional[transforms.Compose] = None) -> transforms.Compose:
    """Crop+flip augmentation shared by Vanilla/GCSC; GCSC inserts RandAugment via `extra`
    (applied after crop/flip, before ToTensor/Normalize, per the assignment).
    """
    ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if extra is not None:
        ops.append(extra)
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return transforms.Compose(ops)


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


def build_cifar10_loaders(
    data_root: str, split_path: str, batch_size: int = 128, train_transform: Optional[transforms.Compose] = None
) -> Dict[str, DataLoader]:
    split = load_cifar10_split(split_path)
    train_tf = train_transform or build_train_transform()
    eval_tf = build_eval_transform()

    train_base = CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    val_base = CIFAR10(root=data_root, train=True, download=True, transform=eval_tf)
    test_set = CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)

    train_set = Subset(train_base, split["train"])
    val_set = Subset(val_base, split["val"])

    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True),
        "val": DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2),
        "test": DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return accuracy(all_labels, all_preds)


def train_vanilla(
    data_root: str,
    split_path: str = "task4/data/cifar10_split_seed6304.json",
    device: str = "cuda",
    batch_size: int = 128,
    epochs: int = 100,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    seed: int = 6304,
    checkpoint_path: str = "task4/results/vanilla/checkpoint.pt",
    metrics_path: str = "task4/results/vanilla/metrics.jsonl",
    train_transform: Optional[transforms.Compose] = None,
) -> nn.Module:
    """Ten-class CIFAR-ResNet-18 trained with plain cross-entropy (Task 4, Step 1).

    `train_transform` is overridden by `task4.methods.gcsc` to insert RandAugment while
    reusing this exact loop/schedule/checkpoint rule, per the assignment's controlled comparison.
    """
    set_seed(seed)  # the spec fixes seed 6304 for every Task 4 training run
    logger = get_logger("task4_vanilla")
    metric_logger = MetricLogger(metrics_path)

    loaders = build_cifar10_loaders(data_root, split_path, batch_size, train_transform)
    model = resnet18_cifar(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        val_acc = evaluate(model, loaders["val"], device)
        metric_logger.log(epoch=epoch, train_loss=epoch_loss / len(loaders["train"]), val_accuracy=val_acc)
        logger.info(f"epoch {epoch}: loss={epoch_loss / len(loaders['train']):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_accuracy": val_acc}, checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    return model
