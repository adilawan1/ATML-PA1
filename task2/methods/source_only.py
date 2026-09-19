from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.logging import MetricLogger, get_logger
from common.metrics import macro_f1
from shared.pacs import PACSDataset, build_pacs_transforms, cycle_loader
from task2.models.backbone import build_resnet18_backbone, freeze_batchnorm_stats

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]


def build_source_loaders(
    protocol: Dict, image_size: int, crop_size: int, batch_per_domain: int, num_workers: int = 2
) -> Dict[str, Dict[str, DataLoader]]:
    """One {"train": loader, "val": loader} pair per source domain, using the shared protocol."""
    loaders: Dict[str, Dict[str, DataLoader]] = {}
    for domain in SOURCE_DOMAINS:
        train_ds = PACSDataset(
            protocol[domain]["train"], transform=build_pacs_transforms(image_size, crop_size, train=True)
        )
        val_ds = PACSDataset(
            protocol[domain]["val"], transform=build_pacs_transforms(image_size, crop_size, train=False)
        )
        loaders[domain] = {
            "train": DataLoader(
                train_ds, batch_size=batch_per_domain, shuffle=True, num_workers=num_workers, drop_last=True
            ),
            "val": DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=num_workers),
        }
    return loaders


@torch.no_grad()
def evaluate_domain(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    from common.metrics import accuracy

    return {"accuracy": accuracy(all_labels, all_preds), "macro_f1": macro_f1(all_labels, all_preds)}


def mean_source_macro_f1(model: nn.Module, val_loaders: Dict[str, DataLoader], device: str) -> float:
    scores = [evaluate_domain(model, val_loaders[d], device)["macro_f1"] for d in SOURCE_DOMAINS]
    return sum(scores) / len(scores)


def train_source_only(
    protocol: Dict,
    device: str = "cuda",
    image_size: int = 256,
    crop_size: int = 224,
    batch_per_domain: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 5,
    checkpoint_path: str = "task2/results/source_only/checkpoint.pt",
    metrics_path: str = "task2/results/source_only/metrics.jsonl",
) -> nn.Module:
    """Cross-entropy ERM over the three labeled source domains with domain-balanced batches.

    This is also the Task 3 ERM baseline -- load the saved checkpoint there rather than
    retraining, per the assignment's shared-baseline requirement.
    """
    logger = get_logger("source_only")
    metric_logger = MetricLogger(metrics_path)

    loaders = build_source_loaders(protocol, image_size, crop_size, batch_per_domain)
    train_iters = {d: cycle_loader(loaders[d]["train"]) for d in SOURCE_DOMAINS}
    steps_per_epoch = max(len(loaders[d]["train"]) for d in SOURCE_DOMAINS)

    model = build_resnet18_backbone(num_classes=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_score = -1.0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        freeze_batchnorm_stats(model)
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            images_list, labels_list = [], []
            for d in SOURCE_DOMAINS:
                images, labels = next(train_iters[d])
                images_list.append(images)
                labels_list.append(labels)
            images = torch.cat(images_list, dim=0).to(device)
            labels = torch.cat(labels_list, dim=0).to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_score = mean_source_macro_f1(model, {d: loaders[d]["val"] for d in SOURCE_DOMAINS}, device)
        metric_logger.log(epoch=epoch, train_loss=epoch_loss / steps_per_epoch, mean_source_val_macro_f1=val_score)
        logger.info(f"epoch {epoch}: loss={epoch_loss / steps_per_epoch:.4f} mean_val_macro_f1={val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            epochs_without_improvement = 0
            import os

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_score": val_score}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"early stopping at epoch {epoch} (best mean_val_macro_f1={best_score:.4f})")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    return model
