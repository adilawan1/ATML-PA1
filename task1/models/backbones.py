from __future__ import annotations

"""Frozen backbone wrappers for Task 1. Every wrapper exposes the same interface:
  - `.normalize`: a torchvision transform applying that backbone's REQUIRED normalization
    (interventions must be built on a common, un-normalized 224x224 RGB tensor first --
    see `task1/data/transforms.py` -- and normalized only here, per-model, at inference time).
  - `.features(x)`: the frozen representation used for the linear head / cosine-stability /
    t-SNE analysis (global-avg-pooled ResNet feature, final ViT class token, or the
    L2-normalized CLIP image embedding).

All three backbones are frozen (`requires_grad_(False)`, `eval()`); a separate linear head
(`task1/models/heads.py`) is trained per backbone on cached features.
"""

from dataclasses import dataclass
from typing import Callable, List

import open_clip
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16


@dataclass
class FrozenBackbone:
    name: str
    model: nn.Module
    normalize: Callable
    feature_dim: int
    features_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor]
    device: str = "cpu"

    def to(self, device: str) -> "FrozenBackbone":
        self.model.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: (B, 3, 224, 224) un-normalized tensor in [0, 1]. Returns (B, feature_dim) on CPU."""
        x = self.normalize(x.to(self.device))
        return self.features_fn(self.model, x).float().cpu()


def extract_features(backbone: FrozenBackbone, images: torch.Tensor, batch_size: int = 100) -> torch.Tensor:
    chunks = [backbone.features(images[i : i + batch_size]) for i in range(0, len(images), batch_size)]
    return torch.cat(chunks)


def _resnet50_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return x.flatten(1)


def build_resnet50() -> FrozenBackbone:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
    return FrozenBackbone("resnet50", model, normalize, feature_dim=2048, features_fn=_resnet50_features)


def _vit_b16_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Replicates torchvision's VisionTransformer.forward up to (not including) `model.heads`,
    # returning the final class token -- the representation the assignment specifies.
    x = model._process_input(x)
    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = model.encoder(x)
    return x[:, 0]


def build_vit_b16() -> FrozenBackbone:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
    return FrozenBackbone("vit_b_16", model, normalize, feature_dim=768, features_fn=_vit_b16_features)


def _clip_image_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    features = model.encode_image(x)
    return features / features.norm(dim=-1, keepdim=True)


def build_clip_vitb32():
    """Returns (FrozenBackbone, tokenizer) for OpenCLIP ViT-B-32 (pretrained='openai')."""
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # `preprocess.transforms` ends with Normalize(mean, std); reuse those exact constants so
    # every backbone still starts from the same common 224x224 RGB tensor.
    normalize = next(t for t in preprocess.transforms if isinstance(t, transforms.Normalize))
    backbone = FrozenBackbone(
        "clip_vitb32", model, normalize, feature_dim=model.visual.output_dim, features_fn=_clip_image_features
    )
    return backbone, tokenizer


@torch.no_grad()
def clip_text_features(backbone: FrozenBackbone, tokenizer, class_names: List[str], template: str = "a photo of a {}.") -> torch.Tensor:
    """L2-normalized text embeddings for the fixed zero-shot prompt (no prompt search)."""
    tokens = tokenizer([template.format(name) for name in class_names]).to(backbone.device)
    text = backbone.model.encode_text(tokens)
    return (text / text.norm(dim=-1, keepdim=True)).float().cpu()


def clip_zero_shot_logits(backbone: FrozenBackbone, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """Scaled image-text cosine similarities; softmax over these gives zero-shot confidence."""
    scale = backbone.model.logit_scale.exp().item()
    return scale * image_features @ text_features.t()
