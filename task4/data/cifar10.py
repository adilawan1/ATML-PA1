from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10

SPLIT_SEED = 6304
DEFAULT_SPLIT_PATH = "task4/data/cifar10_split_seed6304.json"


def build_cifar10_split(data_root: str, out_path: str = DEFAULT_SPLIT_PATH) -> Dict:
    """Stratified 90/10 train/val split of the official CIFAR-10 training partition (seed 6304).
    Checkpoint selection uses only the val split; the full CIFAR-10 test set is used for
    final known-class evaluation and is not touched here.
    """
    train_set = CIFAR10(root=data_root, train=True, download=True)
    targets = train_set.targets
    indices = list(range(len(targets)))

    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=SPLIT_SEED, stratify=targets)
    split = {"seed": SPLIT_SEED, "train": train_idx, "val": val_idx}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f)
    return split


def load_cifar10_split(path: str = DEFAULT_SPLIT_PATH) -> Dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default=DEFAULT_SPLIT_PATH)
    args = parser.parse_args()
    build_cifar10_split(args.data_root, args.out)
    print(f"Wrote CIFAR-10 90/10 split to {args.out}")
