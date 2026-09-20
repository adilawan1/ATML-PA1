from __future__ import annotations

"""Task 1 driver (Steps 1, 2, 4, 5, 6): clean baseline, color bias, translation, patch
shuffle, and representation stability + t-SNE. Cue conflicts (Step 3) live in
`task1/scripts/run_cue_conflicts.py` and reuse the same Evaluator so every model still sees
identical tensors.

Pipeline: cache frozen features for the STL-10 train/val split -> train one linear head per
backbone (AdamW, early stopping) -> evaluate the four predictors (3 heads + zero-shot CLIP) on
the fixed 500-image test subset under each intervention. All numbers land in one JSON
(`task1_results.json`) plus a compact CSV and the figures the report needs.
"""

import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from common.plotting import apply_style, save_figure
from common.seed import set_seed
from task1.analysis.evaluator import (
    BACKBONES,
    PREDICTORS,
    Evaluator,
    average_over_conditions,
    condition_metrics,
    feature_stability,
)
from task1.analysis.representation import draw_projection, fit_tsne, legend_handles
from task1.data.make_subset import build_eval_subset, load_eval_subset
from task1.data.stl10 import load_common_tensors, load_stl10, stratified_train_val_indices
from task1.data.transforms import (
    CARDINAL_DIRECTIONS,
    grayscale,
    hue_rotate,
    patch_shuffle,
    patch_shuffle_generator,
    translate,
)
from task1.models.backbones import (
    build_clip_vitb32,
    build_resnet50,
    build_vit_b16,
    clip_text_features,
    extract_features,
)
from task1.models.heads import train_linear_head


def build_backbones(device: str):
    resnet = build_resnet50().to(device)
    vit = build_vit_b16().to(device)
    clip, tokenizer = build_clip_vitb32()
    clip.to(device)
    return {"resnet50": resnet, "vit_b_16": vit, "clip_vitb32": clip}, tokenizer


def cached_train_val_features(train_set, backbones, cfg, cache_path: str) -> Dict:
    """Frozen features for the stratified 80/20 train/val split of the official train partition,
    extracted in chunks (5000 images at 224x224 would be ~3 GB as one tensor)."""
    if os.path.exists(cache_path):
        print(f"loading cached train/val features from {cache_path}")
        return torch.load(cache_path)

    train_idx, val_idx = stratified_train_val_indices(train_set, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    labels = np.asarray(train_set.labels)
    result = {"train_labels": torch.tensor(labels[train_idx]), "val_labels": torch.tensor(labels[val_idx])}

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


def plot_translation_curve(translation: Dict, deltas, path: str) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for pred in PREDICTORS:
        axes[0].plot(deltas, [translation[str(d)]["mean"][pred]["accuracy"] for d in deltas], marker="o", label=pred)
        axes[1].plot(deltas, [translation[str(d)]["mean"][pred]["consistency"] for d in deltas], marker="o", label=pred)
    axes[0].set_ylabel("accuracy (mean over 4 directions)")
    axes[1].set_ylabel("prediction consistency vs. clean")
    for ax in axes:
        ax.set_xlabel("translation (pixels)")
        ax.set_xticks(list(deltas))
    axes[1].legend(fontsize=8)
    save_figure(fig, path)


def plot_tsne_grid(conditions: Dict, clean_features: Dict, labels: np.ndarray, class_names, cfg: Dict, path: str) -> None:
    """One joint t-SNE fit per (backbone, condition) on clean + transformed features."""
    n = len(labels)
    class_labels = np.concatenate([labels, labels])
    is_transformed = np.concatenate([np.zeros(n, dtype=bool), np.ones(n, dtype=bool)])

    apply_style()
    fig, axes = plt.subplots(len(BACKBONES), len(conditions), figsize=(3.6 * len(conditions), 3.6 * len(BACKBONES)), squeeze=False)
    for row, backbone in enumerate(BACKBONES):
        for col, (cond_name, cond_features) in enumerate(conditions.items()):
            joint = np.concatenate([clean_features[backbone].numpy(), cond_features[backbone].numpy()])
            coords = fit_tsne(joint, perplexity=cfg["representation"]["perplexity"], seed=cfg["representation"]["seed"])
            draw_projection(axes[row][col], coords, class_labels, is_transformed, f"{backbone} | {cond_name}")
            print(f"  t-SNE done: {backbone} | {cond_name}")
    fig.legend(handles=legend_handles(class_names), loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, path)


def compact_table(conditions: Dict) -> pd.DataFrame:
    rows = {}
    for pred in PREDICTORS:
        row = {}
        for cond in ("clean", "grayscale", "hue_rotation", "patch_shuffle"):
            row[f"{cond}_acc"] = conditions[cond][pred]["accuracy"]
            if cond != "clean":
                row[f"{cond}_delta"] = conditions[cond][pred]["acc_delta"]
                row[f"{cond}_consistency"] = conditions[cond][pred]["consistency"]
        rows[pred] = row
    return pd.DataFrame(rows).T


def main(args) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    train_set = load_stl10(args.data_root, "train")
    test_set = load_stl10(args.data_root, "test")
    class_names = list(train_set.classes)

    subset_path = cfg["eval_subset"]["path"]
    if not os.path.exists(subset_path):
        build_eval_subset(args.data_root, subset_path, cfg["eval_subset"]["size"])
    subset = load_eval_subset(subset_path)
    eval_indices, eval_labels = subset["indices"], np.asarray(subset["labels"])

    backbones, tokenizer = build_backbones(args.device)
    text_features = clip_text_features(backbones["clip_vitb32"], tokenizer, class_names, cfg["zero_shot_prompt"])

    features = cached_train_val_features(train_set, backbones, cfg, os.path.join(args.cache_dir, "stl10_train_val_features.pt"))
    heads, head_info = {}, {}
    for name in BACKBONES:
        heads[name], head_info[name] = train_linear_head(
            features["train"][name],
            features["train_labels"],
            features["val"][name],
            features["val_labels"],
            num_classes=len(class_names),
            seed=cfg["seed"],
            **{k: cfg["linear_head"][k] for k in ("lr", "weight_decay", "max_epochs", "patience", "batch_size")},
        )
        print(f"head {name}: {head_info[name]}")

    evaluator = Evaluator(backbones, heads, text_features, cfg["inference_batch_size"])
    clean_images = load_common_tensors(test_set, eval_indices)
    clean_out = evaluator.run(clean_images)

    conditions = {"clean": condition_metrics(clean_out["logits"], eval_labels)}
    stability: Dict = {}
    tsne_features: Dict = {}

    def run_static(name: str, images: torch.Tensor) -> Dict:
        out = evaluator.run(images)
        conditions[name] = condition_metrics(out["logits"], eval_labels, clean_out["logits"])
        stability[name] = feature_stability(clean_out["features"], out["features"])
        return out

    iv = cfg["interventions"]
    tsne_features["grayscale"] = run_static("grayscale", grayscale(clean_images))["features"]
    run_static("hue_rotation", hue_rotate(clean_images, iv["hue_shift"]))

    generator = patch_shuffle_generator(iv["patch_shuffle_seed"])
    shuffled = torch.stack([patch_shuffle(x, iv["patch_grid"], generator) for x in clean_images])
    tsne_features["patch shuffle"] = run_static("patch_shuffle", shuffled)["features"]

    rep = cfg["representation"]
    translation, translation_stability = {}, {}
    for delta in iv["translation_pixels"]:
        per_dir_metrics, per_dir_stability = {}, {}
        directions = {"none": (0, 0)} if delta == 0 else CARDINAL_DIRECTIONS
        for direction, (dx, dy) in directions.items():
            out = clean_out if delta == 0 else evaluator.run(translate(clean_images, dx * delta, dy * delta))
            per_dir_metrics[direction] = condition_metrics(out["logits"], eval_labels, clean_out["logits"])
            per_dir_stability[direction] = feature_stability(clean_out["features"], out["features"])
            if delta == rep["translation_delta"] and direction == rep["translation_direction"]:
                tsne_features[f"translation {delta}px {direction}"] = out["features"]
        translation[str(delta)] = {"mean": average_over_conditions(per_dir_metrics), "per_direction": per_dir_metrics}
        translation_stability[str(delta)] = {
            "mean": average_over_conditions(per_dir_stability),
            "per_direction": per_dir_stability,
        }
        print(f"translation {delta}px done")
    stability["translation"] = translation_stability

    plot_translation_curve(translation, iv["translation_pixels"], os.path.join(args.fig_dir, "task1_translation_curve.png"))

    if not args.skip_tsne:
        plot_tsne_grid(tsne_features, clean_out["features"], eval_labels, class_names, cfg, os.path.join(args.fig_dir, "task1_tsne.png"))

    table = compact_table(conditions)
    table.to_csv(os.path.join(args.out_dir, "compact_comparison.csv"))
    print("\n" + table.round(3).to_string())

    results = {
        "meta": {
            "torch": torch.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "eval_subset": subset_path,
            "n_eval": len(eval_indices),
        },
        "config": cfg,
        "class_names": class_names,
        "heads": head_info,
        "conditions": conditions,
        "translation": translation,
        "representation_stability": stability,
        "tsne": {"method": "t-SNE", "perplexity": rep["perplexity"], "seed": rep["seed"], "init": "pca", "conditions": list(tsne_features)},
    }
    with open(os.path.join(args.out_dir, "task1_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 'task1_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="task1/configs/task1.yaml")
    parser.add_argument("--out-dir", default="task1/results")
    parser.add_argument("--fig-dir", default="report/figures")
    parser.add_argument("--cache-dir", default="task1/cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-tsne", action="store_true")
    main(parser.parse_args())
