from __future__ import annotations

"""Shared Task 1 setup used by every driver script: frozen backbones -> cached train/val
features -> linear heads (trained once, then cached) -> Evaluator. Both `run_task1.py` and
`run_cue_conflicts.py` go through `load_stack`, so all conditions are scored by the identical
heads on identical features.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from task1.analysis.evaluator import BACKBONES, Evaluator
from task1.data.stl10 import load_common_tensors, load_stl10, stratified_train_val_indices
from task1.models.backbones import (
    build_clip_vitb32,
    build_resnet50,
    build_vit_b16,
    clip_text_features,
    extract_features,
)
from task1.models.heads import train_linear_head


@dataclass
class Task1Stack:
    train_set: object
    test_set: object
    class_names: List[str]
    backbones: Dict
    heads: Dict[str, nn.Linear]
    head_info: Dict[str, Dict]
    evaluator: Evaluator


def build_backbones(device: str):
    resnet = build_resnet50().to(device)
    vit = build_vit_b16().to(device)
    clip, tokenizer = build_clip_vitb32()
    clip.to(device)
    return {"resnet50": resnet, "vit_b_16": vit, "clip_vitb32": clip}, tokenizer


def cached_train_val_features(train_set, backbones, cfg: Dict, cache_path: str) -> Dict:
    """Frozen features for the stratified 80/20 train/val split of the official train partition,
    extracted in chunks (5000 images at 224x224 would be ~3 GB as one tensor)."""
    if os.path.exists(cache_path):
        print(f"loading cached train/val features from {cache_path}")
        return torch.load(cache_path)

    train_idx, val_idx = stratified_train_val_indices(train_set, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    labels = np.asarray(train_set.labels)
    result = {
        "train_labels": torch.tensor(labels[train_idx], dtype=torch.long),
        "val_labels": torch.tensor(labels[val_idx], dtype=torch.long),
    }
    for split_name, indices in [("train", train_idx), ("val", val_idx)]:
        per_backbone = {name: [] for name in backbones}
        chunk = 250
        for start in range(0, len(indices), chunk):
            images = load_common_tensors(train_set, indices[start : start + chunk])
            for name, bb in backbones.items():
                per_backbone[name].append(extract_features(bb, images, cfg["inference_batch_size"]))
            print(f"  {split_name}: {min(start + chunk, len(indices))}/{len(indices)}")
        result[split_name] = {name: torch.cat(chunks) for name, chunks in per_backbone.items()}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(result, cache_path)
    return result


def train_or_load_heads(features: Dict, cfg: Dict, num_classes: int, cache_dir: str) -> Tuple[Dict, Dict]:
    """One linear head per backbone. Trained once and cached with the settings that produced
    it, so a later script gets the exact same heads (a changed config retrains)."""
    path = os.path.join(cache_dir, "linear_heads.pt")
    signature = {"seed": cfg["seed"], **cfg["linear_head"]}
    if os.path.exists(path):
        saved = torch.load(path)
        if saved["signature"] == signature:
            heads = {}
            for name, state in saved["states"].items():
                head = nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
                head.load_state_dict(state)
                heads[name] = head.eval()
            print(f"loaded cached linear heads from {path}")
            return heads, saved["info"]

    heads, info = {}, {}
    for name in BACKBONES:
        heads[name], info[name] = train_linear_head(
            features["train"][name],
            features["train_labels"],
            features["val"][name],
            features["val_labels"],
            num_classes=num_classes,
            seed=cfg["seed"],
            **{k: cfg["linear_head"][k] for k in ("lr", "weight_decay", "max_epochs", "patience", "batch_size")},
        )
        print(f"head {name}: {info[name]}")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"signature": signature, "states": {n: h.state_dict() for n, h in heads.items()}, "info": info}, path)
    return heads, info


def save_split_indices(train_set, cfg: Dict, path: str = "task1/data/stl10_train_val_split_seed6304.json") -> None:
    """The stratified 80/20 train/val split of the official STL-10 training partition, saved as indices
    (deterministic, seed 6304) so it is part of the repository rather than only implied by code."""
    import json

    train_idx, val_idx = stratified_train_val_indices(train_set, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": cfg["split"]["seed"], "val_fraction": cfg["split"]["val_fraction"], "train": train_idx, "val": val_idx}, f)


def load_stack(data_root: str, cfg: Dict, cache_dir: str, device: str) -> Task1Stack:
    train_set = load_stl10(data_root, "train")
    save_split_indices(train_set, cfg)
    test_set = load_stl10(data_root, "test")
    class_names = list(train_set.classes)

    backbones, tokenizer = build_backbones(device)
    text_features = clip_text_features(backbones["clip_vitb32"], tokenizer, class_names, cfg["zero_shot_prompt"])
    features = cached_train_val_features(train_set, backbones, cfg, os.path.join(cache_dir, "stl10_train_val_features.pt"))
    heads, head_info = train_or_load_heads(features, cfg, len(class_names), cache_dir)

    evaluator = Evaluator(backbones, heads, text_features, cfg["inference_batch_size"])
    return Task1Stack(train_set, test_set, class_names, backbones, heads, head_info, evaluator)
