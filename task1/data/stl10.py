from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10

from task1.data.transforms import to_common_image

SPLIT_SEED = 6304


def load_stl10(data_root: str, split: str) -> STL10:
    """Official STL-10 partition ('train' = 5000 labeled images, 'test' = 8000). Returns PIL
    images (no transform) so every model sees the same `to_common_image` 224x224 tensor.
    """
    return STL10(root=data_root, split=split, download=True)


def stratified_train_val_indices(train_set: STL10, val_fraction: float = 0.2, seed: int = SPLIT_SEED) -> Tuple[List[int], List[int]]:
    """Stratified 80/20 split of the OFFICIAL training partition (seed 6304)."""
    labels = np.asarray(train_set.labels)
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_fraction, random_state=seed, stratify=labels)
    return sorted(train_idx.tolist()), sorted(val_idx.tolist())


def load_common_tensors(dataset: STL10, indices: List[int]) -> torch.Tensor:
    """(N, 3, 224, 224) un-normalized float tensor in [0, 1] for the given dataset indices."""
    return torch.stack([to_common_image(dataset[i][0]) for i in indices])
