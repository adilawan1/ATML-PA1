from __future__ import annotations

"""Shared evaluation machinery for every Task 1 condition (clean, color, translation, patch
shuffle, cue conflicts): one forward pass per backbone per image set, giving both the four
predictors' logits (three trained linear heads + zero-shot CLIP) and the frozen features used
by the representation analysis. Keeping this in one place guarantees every model receives the
identical tensors and that every metric is computed the same way.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from common.metrics import accuracy, consistency, macro_f1, mean_max_confidence, per_class_accuracy
from task1.analysis.feature_similarity import cosine_stability
from task1.models.backbones import FrozenBackbone, clip_zero_shot_logits, extract_features

BACKBONES = ("resnet50", "vit_b_16", "clip_vitb32")
PREDICTORS = ("resnet50", "vit_b_16", "clip_head", "clip_zeroshot")


class Evaluator:
    def __init__(
        self,
        backbones: Dict[str, FrozenBackbone],
        heads: Dict[str, nn.Linear],
        text_features: torch.Tensor,
        batch_size: int = 100,
    ):
        self.backbones = backbones
        self.heads = heads
        self.text_features = text_features
        self.batch_size = batch_size

    @torch.no_grad()
    def run(self, images: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """`images`: (N, 3, 224, 224) un-normalized tensor in [0, 1]."""
        feats = {name: extract_features(bb, images, self.batch_size) for name, bb in self.backbones.items()}
        logits = {
            "resnet50": self.heads["resnet50"](feats["resnet50"]),
            "vit_b_16": self.heads["vit_b_16"](feats["vit_b_16"]),
            "clip_head": self.heads["clip_vitb32"](feats["clip_vitb32"]),
            "clip_zeroshot": clip_zero_shot_logits(self.backbones["clip_vitb32"], feats["clip_vitb32"], self.text_features),
        }
        return {"features": feats, "logits": logits}


def condition_metrics(
    logits: Dict[str, torch.Tensor], labels: np.ndarray, clean_logits: Dict[str, torch.Tensor] = None
) -> Dict[str, Dict]:
    """Per-predictor accuracy / macro-F1 / mean max confidence / per-class accuracy. When
    `clean_logits` is given (same images, un-transformed), also the accuracy change and the
    prediction consistency relative to the model's own clean predictions.
    """
    out = {}
    for name, lg in logits.items():
        preds = lg.argmax(dim=1).numpy()
        metrics = {
            "accuracy": accuracy(labels, preds),
            "macro_f1": macro_f1(labels, preds),
            "mean_max_conf": mean_max_confidence(lg.softmax(dim=1).numpy()),
            "per_class_accuracy": [float(v) for v in per_class_accuracy(labels, preds, lg.shape[1])],
        }
        if clean_logits is not None:
            clean_preds = clean_logits[name].argmax(dim=1).numpy()
            metrics["acc_delta"] = metrics["accuracy"] - accuracy(labels, clean_preds)
            metrics["consistency"] = consistency(clean_preds, preds)
        out[name] = metrics
    return out


def feature_stability(clean_features: Dict[str, torch.Tensor], cond_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """I_T per backbone: mean cosine similarity between paired clean/transformed features."""
    return {name: cosine_stability(clean_features[name], cond_features[name]) for name in clean_features}


def average_over_conditions(per_condition: Dict[str, Dict]) -> Dict:
    """Element-wise mean of identically-structured metric dicts (e.g. across 4 directions).
    Works for `{predictor: {metric: scalar-or-list}}` and `{backbone: scalar}` alike.
    """
    names = list(per_condition)
    first = per_condition[names[0]]
    out = {}
    for key, value in first.items():
        if isinstance(value, dict):
            out[key] = average_over_conditions({n: per_condition[n][key] for n in names})
        else:
            out[key] = np.mean(np.array([per_condition[n][key] for n in names], dtype=float), axis=0).tolist()
    return out
