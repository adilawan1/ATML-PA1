from __future__ import annotations

"""In-memory PACS for Tasks 2 and 3.

Every image is decoded once and resized to 256x256 (exactly the spec's Resize step), kept as one
uint8 tensor, and turned into training batches by seeded random 224x224 crops + horizontal flips
(eval: center 224 crop), then ImageNet-normalized. This is the same preprocessing as a
torchvision DataLoader pipeline, without the per-step JPEG decoding that would otherwise bottleneck
Colab's few CPU cores.

Leakage guards, by construction:
  * `include_target=False` (Task 3 training/selection/diagnostics) never opens a Sketch file and
    its cache contains only source images.
  * `labels()` refuses the target domain; only `target_labels_for_final_eval()` returns Sketch
    labels, and only the final evaluation scripts call it.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
IMAGE_SIZE = 256
CROP_SIZE = 224

_transform_meta = ResNet18_Weights.IMAGENET1K_V1.transforms()
_MEAN = torch.tensor(_transform_meta.mean).view(1, 3, 1, 1)
_STD = torch.tensor(_transform_meta.std).view(1, 3, 1, 1)
_RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))


def _load_resized(path: str) -> torch.Tensor:
    image = _RESIZE(Image.open(path).convert("RGB"))
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)  # uint8 (3, H, W)


class BatchStream:
    """Endless stream of `batch_size` indices into a set of size `n`: consecutive random
    permutations, so a small domain simply cycles ('cycle a loader when necessary')."""

    def __init__(self, n: int, batch_size: int, generator: torch.Generator):
        self.n, self.batch_size, self.generator = n, batch_size, generator
        self.buffer = torch.empty(0, dtype=torch.long)

    def next(self) -> torch.Tensor:
        while len(self.buffer) < self.batch_size:
            self.buffer = torch.cat([self.buffer, torch.randperm(self.n, generator=self.generator)])
        out, self.buffer = self.buffer[: self.batch_size], self.buffer[self.batch_size :]
        return out


class PACSData:
    def __init__(self, protocol: Dict, pacs_root: str, cache_path: Optional[str] = None, include_target: bool = False):
        self._rows: Dict[Tuple[str, str], torch.Tensor] = {}
        self._labels: Dict[Tuple[str, str], torch.Tensor] = {}
        paths: List[str] = []

        def register(key: Tuple[str, str], samples) -> None:
            start = len(paths)
            paths.extend(p for p, _ in samples)
            self._rows[key] = torch.arange(start, len(paths))
            self._labels[key] = torch.tensor([label for _, label in samples], dtype=torch.long)

        for domain in SOURCE_DOMAINS:
            for split in ("train", "val"):
                register((domain, split), protocol[domain][split])
        if include_target:
            register((TARGET_DOMAIN, "all"), protocol[TARGET_DOMAIN]["all"])
        self.include_target = include_target
        self.images = self._load_images(paths, pacs_root, cache_path)

    @staticmethod
    def _load_images(paths: List[str], pacs_root: str, cache_path: Optional[str]) -> torch.Tensor:
        if cache_path and os.path.exists(cache_path):
            saved = torch.load(cache_path)
            if saved["paths"] == paths:
                print(f"loaded {len(paths)} preprocessed PACS images from {cache_path}")
                return saved["images"]
        print(f"decoding {len(paths)} PACS images (one-time, a minute or two) ...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            images = torch.stack(list(pool.map(_load_resized, [os.path.join(pacs_root, p) for p in paths])))
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save({"paths": paths, "images": images}, cache_path)
        return images

    # ---- index / label access ------------------------------------------------------------
    def rows(self, domain: str, split: str) -> torch.Tensor:
        return self._rows[(domain, split)]

    def labels(self, domain: str, split: str) -> torch.Tensor:
        if domain == TARGET_DOMAIN:
            raise PermissionError("target labels are only available via target_labels_for_final_eval()")
        return self._labels[(domain, split)]

    def target_labels_for_final_eval(self) -> torch.Tensor:
        return self._labels[(TARGET_DOMAIN, "all")]

    # ---- batches -------------------------------------------------------------------------
    @staticmethod
    def _normalize(batch_u8: torch.Tensor) -> torch.Tensor:
        return (batch_u8.float() / 255.0 - _MEAN.to(batch_u8.device)) / _STD.to(batch_u8.device)

    def batch_train(self, rows: torch.Tensor, generator: torch.Generator, device: str) -> torch.Tensor:
        """Random 224 crop + horizontal flip (per image, seeded), normalized, on `device`."""
        x = self.images[rows]
        n = x.shape[0]
        span = IMAGE_SIZE - CROP_SIZE + 1
        tops = torch.randint(0, span, (n,), generator=generator).tolist()
        lefts = torch.randint(0, span, (n,), generator=generator).tolist()
        flips = (torch.rand(n, generator=generator) < 0.5).tolist()
        out = torch.empty(n, 3, CROP_SIZE, CROP_SIZE, dtype=torch.uint8)
        for i in range(n):
            patch = x[i, :, tops[i] : tops[i] + CROP_SIZE, lefts[i] : lefts[i] + CROP_SIZE]
            out[i] = patch.flip(-1) if flips[i] else patch
        return self._normalize(out.to(device))

    def batch_eval(self, rows: torch.Tensor, device: str) -> torch.Tensor:
        """Center 224 crop, normalized, on `device`."""
        offset = (IMAGE_SIZE - CROP_SIZE) // 2
        x = self.images[rows][:, :, offset : offset + CROP_SIZE, offset : offset + CROP_SIZE]
        return self._normalize(x.to(device))
