from __future__ import annotations

"""Task 1 controlled interventions. Every transform operates on a common, UN-normalized
224x224 RGB tensor in [0, 1] (`to_common_image`) so all three backbones see identical pixels;
per-model normalization happens only inside `task1.models.backbones.FrozenBackbone.features`.
"""

from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
PATCH_SHUFFLE_SEED = 6304


def to_common_image(image: Image.Image) -> torch.Tensor:
    resize = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    return resize(image.convert("RGB"))


def grayscale(x: torch.Tensor) -> torch.Tensor:
    """Removes chromatic information, preserves geometry (Step 2's required intervention)."""
    return TF.rgb_to_grayscale(x, num_output_channels=3)


def hue_rotate(x: torch.Tensor, hue_shift: float = 0.3) -> torch.Tensor:
    """Fixed hue rotation (Step 2's chosen additional color transform): shifts chromatic
    identity while preserving luminance/geometry -- tests sensitivity to changed, not removed,
    color. `hue_shift` in [-0.5, 0.5].
    """
    return TF.adjust_hue(x, hue_shift)


def translate(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Reflection-pad then shifted-crop translation by (dx, dy) pixels (Step 4). (0, 0) is
    the identity/no-shift case used as the delta=0 point on the consistency/accuracy curve.
    """
    pad = max(abs(dx), abs(dy))
    padded = TF.pad(x, [pad, pad, pad, pad], padding_mode="reflect")
    top = pad - dy
    left = pad - dx
    return TF.crop(padded, top=top, left=left, height=x.shape[-2], width=x.shape[-1])


CARDINAL_DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def translate_all_directions(x: torch.Tensor, magnitude: int) -> dict:
    """Translate by `magnitude` pixels in each of the 4 cardinal directions; average the
    resulting metric across directions per the assignment's translation-curve protocol.
    """
    return {
        direction: translate(x, dx=dx * magnitude, dy=dy * magnitude)
        for direction, (dx, dy) in CARDINAL_DIRECTIONS.items()
    }


def patch_shuffle(x: torch.Tensor, grid: int = 4, generator: torch.Generator = None) -> torch.Tensor:
    """One non-identity `grid`x`grid` pixel-space patch permutation (Step 5). Pass a
    `torch.Generator` seeded with 6304 so the SAME shuffled image is reused across models.
    """
    c, h, w = x.shape
    ph, pw = h // grid, w // grid
    patches = x.unfold(1, ph, ph).unfold(2, pw, pw)  # (c, grid, grid, ph, pw)
    patches = patches.contiguous().view(c, grid * grid, ph, pw).permute(1, 0, 2, 3)  # (grid*grid, c, ph, pw)

    n = grid * grid
    perm = torch.randperm(n, generator=generator)
    if torch.equal(perm, torch.arange(n)):
        perm = torch.roll(perm, shifts=1)  # guarantee non-identity

    shuffled = patches[perm]
    rows = [torch.cat(list(shuffled[r * grid : (r + 1) * grid]), dim=2) for r in range(grid)]
    return torch.cat(rows, dim=1)


def patch_shuffle_generator(seed: int = PATCH_SHUFFLE_SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
