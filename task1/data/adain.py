from __future__ import annotations

"""Thin wrapper around the public pytorch-AdaIN implementation (github.com/naoto0804/pytorch-AdaIN,
MIT licence; method: Huang & Belongie, 2017). The repository is cloned at runtime into
`third_party/` (git-ignored) and pinned to a commit; its `net.decoder` / `net.vgg` definitions
and the released `decoder.pth` / `vgg_normalised.pth` weights are used unmodified. Only the short
`style_transfer` routine from its `test.py` (encode both images, AdaIN on relu4_1 features,
blend by alpha, decode) is re-stated here so it can run on batches.
"""

import importlib
import os
import subprocess
import sys
import urllib.request

import torch
import torch.nn as nn

REPO_URL = "https://github.com/naoto0804/pytorch-AdaIN.git"
REPO_COMMIT = "47950d0e6656a95a80a4b105c4c0f58d38ef785c"
WEIGHTS_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/{}"
WEIGHT_FILES = ("decoder.pth", "vgg_normalised.pth")
DEFAULT_REPO_DIR = "third_party/pytorch-AdaIN"


def ensure_adain(repo_dir: str = DEFAULT_REPO_DIR) -> str:
    """Clone the repo at the pinned commit and fetch the two weight files if missing."""
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        os.makedirs(os.path.dirname(os.path.abspath(repo_dir)), exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO_URL, repo_dir], check=True)
    subprocess.run(["git", "-C", repo_dir, "checkout", "-q", REPO_COMMIT], check=True)

    models_dir = os.path.join(repo_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    for name in WEIGHT_FILES:
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(WEIGHTS_URL.format(name), path + ".part")
            os.replace(path + ".part", path)
    return repo_dir


def _import_upstream(repo_dir: str):
    # upstream's net.py does `from function import ...`, so its directory must be importable
    path = os.path.abspath(repo_dir)
    sys.path.insert(0, path)
    try:
        return importlib.import_module("net"), importlib.import_module("function")
    finally:
        sys.path.remove(path)


class AdaINStylizer:
    def __init__(self, device: str = "cpu", repo_dir: str = DEFAULT_REPO_DIR):
        repo_dir = ensure_adain(repo_dir)
        net, function = _import_upstream(repo_dir)
        decoder, vgg = net.decoder, net.vgg
        decoder.load_state_dict(torch.load(os.path.join(repo_dir, "models", "decoder.pth"), map_location="cpu"))
        vgg.load_state_dict(torch.load(os.path.join(repo_dir, "models", "vgg_normalised.pth"), map_location="cpu"))

        self.device = device
        self.encoder = nn.Sequential(*list(vgg.children())[:31]).to(device).eval().requires_grad_(False)  # up to relu4_1
        self.decoder = decoder.to(device).eval().requires_grad_(False)
        self._adain = function.adaptive_instance_normalization

    @torch.no_grad()
    def stylize(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0, batch_size: int = 20) -> torch.Tensor:
        """`content`, `style`: (N, 3, H, W) in [0, 1] (H, W multiples of 8). Returns (N, 3, H, W)
        in [0, 1] with the content's layout and the style's feature statistics; `alpha` in
        [0, 1] is the stylization strength (1 = full AdaIN).
        """
        assert 0.0 <= alpha <= 1.0 and content.shape == style.shape
        outputs = []
        for start in range(0, len(content), batch_size):
            c = content[start : start + batch_size].to(self.device)
            s = style[start : start + batch_size].to(self.device)
            content_feat, style_feat = self.encoder(c), self.encoder(s)
            feat = alpha * self._adain(content_feat, style_feat) + (1.0 - alpha) * content_feat
            outputs.append(self.decoder(feat).clamp(0.0, 1.0).cpu())
        return torch.cat(outputs)
