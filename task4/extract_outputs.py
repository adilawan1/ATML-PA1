from __future__ import annotations

"""Freeze a trained checkpoint and cache its penultimate features + logits for CIFAR-10
train/val/test and both CIFAR-100 unknown groups, so every downstream score (MSP/MLS/Energy/
Mahalanobis) and `evaluate_osr.py` read identical, already-computed outputs.

Train features are cached WITHOUT augmentation (eval transform), per the assignment's
Mahalanobis requirement ("estimate class means ... from unaugmented CIFAR-10 training
features"). All splits use the same eval transform for consistency.
"""

import argparse
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from task4.data.cifar10 import load_cifar10_split
from task4.data.cifar100_unknowns import CIFAR100UnknownSubset
from task4.methods.vanilla import build_eval_transform
from task4.models.resnet_cifar import penultimate_features, resnet18_cifar


@torch.no_grad()
def _extract(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, torch.Tensor]:
    model.eval()
    all_features, all_logits, all_labels = [], [], []
    for images, labels in loader:
        images = images.to(device)
        feats = penultimate_features(model, images)
        logits = model.fc(feats)
        all_features.append(feats.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return {
        "features": torch.cat(all_features),
        "logits": torch.cat(all_logits),
        "labels": torch.cat(all_labels),
    }


def extract_all_outputs(
    checkpoint_path: str,
    data_root: str,
    method_name: str,
    split_path: str = "task4/data/cifar10_split_seed6304.json",
    cache_dir: str = "task4/cache",
    device: str = "cuda",
    batch_size: int = 256,
) -> Dict[str, str]:
    model = resnet18_cifar(num_classes=10).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])

    eval_tf = build_eval_transform()
    split = load_cifar10_split(split_path)

    train_base = CIFAR10(root=data_root, train=True, download=True, transform=eval_tf)
    test_set = CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)
    train_set = Subset(train_base, split["train"])  # unaugmented, for Mahalanobis class stats
    val_set = Subset(train_base, split["val"])

    near_set = CIFAR100UnknownSubset(data_root, group="near", transform=eval_tf)
    far_set = CIFAR100UnknownSubset(data_root, group="far", transform=eval_tf)

    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
        "near": DataLoader(near_set, batch_size=batch_size, shuffle=False, num_workers=2),
        "far": DataLoader(far_set, batch_size=batch_size, shuffle=False, num_workers=2),
    }

    os.makedirs(cache_dir, exist_ok=True)
    saved_paths = {}
    for split_name, loader in loaders.items():
        outputs = _extract(model, loader, device)
        path = os.path.join(cache_dir, f"{method_name}_{split_name}.pt")
        torch.save(outputs, path)
        saved_paths[split_name] = path
        print(f"{method_name}/{split_name}: features={tuple(outputs['features'].shape)} -> {path}")

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--method-name", required=True, choices=["vanilla", "gcsc", "proser"])
    parser.add_argument("--split-path", default="task4/data/cifar10_split_seed6304.json")
    parser.add_argument("--cache-dir", default="task4/cache")
    args = parser.parse_args()
    extract_all_outputs(args.checkpoint, args.data_root, args.method_name, args.split_path, args.cache_dir)
