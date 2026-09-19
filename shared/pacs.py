from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

_IMAGENET_MEAN = ResNet18_Weights.IMAGENET1K_V1.transforms().mean
_IMAGENET_STD = ResNet18_Weights.IMAGENET1K_V1.transforms().std


def build_pacs_transforms(image_size: int = 256, crop_size: int = 224, train: bool = True):
    """Shared Task 2/3 preprocessing: resize-then-crop with ImageNet normalization.

    Train: resize to `image_size`, random `crop_size` crop, random horizontal flip.
    Eval: resize to `image_size`, center `crop_size` crop. No other augmentation, per spec.
    """
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def cycle_loader(loader):
    """Yield batches from `loader` forever, reshuffling each pass -- used to keep a smaller
    domain's loader supplying batches for as many steps as the largest domain/epoch needs.
    """
    while True:
        for batch in loader:
            yield batch


def list_domain_files(root: str, domain: str) -> List[Tuple[str, int]]:
    """Return sorted (path, label) pairs for one PACS domain.

    Sorting makes the listing deterministic across OSes/filesystems before any seeded split
    is applied on top of it.
    """
    domain_dir = Path(root) / domain
    items: List[Tuple[str, int]] = []
    for cls in CLASSES:
        cls_dir = domain_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class folder {cls_dir}. Check --root points at the PACS directory "
                f"whose immediate subfolders are {DOMAINS}."
            )
        for img_path in sorted(cls_dir.glob("*")):
            items.append((str(img_path), CLASS_TO_IDX[cls]))
    return items


class PACSDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[Callable] = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
