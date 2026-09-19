from __future__ import annotations

from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

# Fixed grouping (Task 4 spec) -- may not be revised after seeing results.
NEAR_UNKNOWN = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_UNKNOWN = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]


class CIFAR100UnknownSubset(Dataset):
    """All CIFAR-100 TEST images belonging to a fixed group of 8 fine classes (100 images/class,
    800 total per group). CIFAR-100 TRAIN images must never be used anywhere in Task 4.
    """

    def __init__(self, data_root: str, group: str, transform: Optional[object] = None):
        if group not in ("near", "far"):
            raise ValueError("group must be 'near' or 'far'")
        wanted_names = NEAR_UNKNOWN if group == "near" else FAR_UNKNOWN

        base = CIFAR100(root=data_root, train=False, download=True)
        name_to_idx = {name: idx for idx, name in enumerate(base.classes)}
        wanted_idx = {name_to_idx[name] for name in wanted_names}

        self.samples = [(img, target) for img, target in zip(base.data, base.targets) if target in wanted_idx]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_array, _ = self.samples[idx]
        image = Image.fromarray(img_array)
        if self.transform is not None:
            image = self.transform(image)
        return image, -1  # unknown: not a valid CIFAR-10 label
