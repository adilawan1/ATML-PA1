from __future__ import annotations

"""Evaluate the four predictors on the frozen cue-conflict set written by
`task1/data/make_cue_conflicts.py` (Task 1, Step 3, plus its Step 6 representation analysis).

Each prediction is labelled shape (= content class), texture (= style class) or other, giving
Shape Bias(%) = N_shape / (N_shape + N_texture) * 100 and Coverage(%) = (N_shape + N_texture) / N_total * 100.
Report both: a high shape bias with tiny coverage is weak evidence. Also computes, per backbone, the
cosine stability between each conflict image and its clean content counterpart, a joint t-SNE, and a
small set of agreement / disagreement / failure examples with every model's prediction.
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

from common.metrics import accuracy
from common.plotting import apply_style, save_figure
from common.seed import set_seed
from task1.analysis.evaluator import BACKBONES, PREDICTORS, feature_stability
from task1.analysis.evaluate_bias import classify_cue_conflict_predictions, shape_bias_and_coverage
from task1.analysis.representation import draw_projection, fit_tsne, legend_handles
from task1.analysis.setup import load_stack
from task1.data.stl10 import load_common_tensors

DECISION_NAMES = {0: "shape", 1: "texture", 2: "other"}


def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def plot_tsne(clean_features: Dict, conflict_features: Dict, shape_labels: np.ndarray, class_names, cfg: Dict, path: str) -> None:
    n = len(shape_labels)
    class_labels = np.concatenate([shape_labels, shape_labels])
    is_transformed = np.concatenate([np.zeros(n, dtype=bool), np.ones(n, dtype=bool)])
    apply_style()
    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(3.6 * len(BACKBONES), 3.9), squeeze=False)
    for col, backbone in enumerate(BACKBONES):
        joint = np.concatenate([clean_features[backbone].numpy(), conflict_features[backbone].numpy()])
        coords = fit_tsne(joint, perplexity=cfg["representation"]["perplexity"], seed=cfg["representation"]["seed"])
        draw_projection(axes[0][col], coords, class_labels, is_transformed, f"{backbone} | cue conflict (color = shape class)")
    fig.legend(handles=legend_handles(class_names), loc="lower center", ncol=6, fontsize=7)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save_figure(fig, path)


def pick_examples(codes: Dict[str, np.ndarray], per_category: int = 2) -> Dict[str, list]:
    """Deterministic (lowest index first) examples of: every predictor chose shape, every predictor
    chose texture, predictors split between shape and texture, and every predictor chose neither."""
    stacked = np.stack([codes[p] for p in PREDICTORS])  # (n_predictors, N)
    masks = {
        "all_shape": (stacked == 0).all(0),
        "all_texture": (stacked == 1).all(0),
        "disagree_shape_vs_texture": (stacked == 0).any(0) & (stacked == 1).any(0),
        "all_other": (stacked == 2).all(0),
    }
    return {name: np.flatnonzero(mask)[:per_category].tolist() for name, mask in masks.items()}


def plot_examples(examples, images, items, preds, codes, class_names, path: str) -> None:
    picks = [(cat, i) for cat, idxs in examples.items() for i in idxs]
    if not picks:
        return
    cols = 4
    rows = -(-len(picks) // cols)
    apply_style()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 4.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (cat, i) in enumerate(picks):
        ax = axes[k // cols][k % cols]
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        item = items[i]
        lines = [f"[{cat}]", f"shape={class_names[item['shape_label']]}  texture={class_names[item['texture_label']]}"]
        lines += [f"{p}: {class_names[preds[p][i]]} ({DECISION_NAMES[codes[p][i]]})" for p in PREDICTORS]
        ax.set_title("\n".join(lines), fontsize=6.5, loc="left")
    fig.tight_layout(h_pad=2.5)
    save_figure(fig, path)


def main(args) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    with open(args.meta) as f:
        meta = json.load(f)
    items = meta["kept"]
    images = torch.load(os.path.join(args.cache_dir, "cue_conflicts.pt"))["images"].float() / 255.0
    assert len(items) == len(images), "metadata and saved images are out of sync -- regenerate the conflicts"
    shape_labels = np.array([i["shape_label"] for i in items])
    texture_labels = np.array([i["texture_label"] for i in items])

    stack = load_stack(args.data_root, cfg, args.cache_dir, args.device)
    class_names = stack.class_names
    out = stack.evaluator.run(images)
    clean_out = stack.evaluator.run(load_common_tensors(stack.test_set, [i["content_idx"] for i in items]))

    preds = {p: out["logits"][p].argmax(1).numpy() for p in PREDICTORS}
    clean_preds = {p: clean_out["logits"][p].argmax(1).numpy() for p in PREDICTORS}
    codes = {p: classify_cue_conflict_predictions(preds[p], shape_labels, texture_labels) for p in PREDICTORS}

    groups = sorted({(i["pair"], i["direction"]) for i in items})
    results = {"n_conflicts": len(items), "totals": meta["totals"], "rule": meta["rule_description"], "predictors": {}}
    rows = {}
    for p in PREDICTORS:
        overall = shape_bias_and_coverage(codes[p])
        by_cell = {}
        for pair, direction in groups:
            mask = np.array([(i["pair"], i["direction"]) == (pair, direction) for i in items])
            by_cell[f"{pair} | {direction}"] = shape_bias_and_coverage(codes[p][mask])
        results["predictors"][p] = {
            **overall,
            "clean_content_accuracy": accuracy(shape_labels, clean_preds[p]),
            "by_pair_direction": by_cell,
        }
        rows[p] = {k: overall[k] for k in ("n_shape", "n_texture", "n_other", "shape_bias_pct", "coverage_pct")}
        rows[p]["clean_content_acc"] = results["predictors"][p]["clean_content_accuracy"]

    results["representation_stability"] = feature_stability(clean_out["features"], out["features"])
    results["examples"] = pick_examples(codes)

    os.makedirs(args.out_dir, exist_ok=True)
    table = pd.DataFrame(rows).T
    table.to_csv(os.path.join(args.out_dir, "cue_conflicts_summary.csv"))
    print(table.round(2).to_string())
    print("\ncosine stability (clean content vs. conflict):", results["representation_stability"])
    with open(os.path.join(args.out_dir, "cue_conflicts.json"), "w") as f:
        json.dump(clean_nans(results), f, indent=2)

    plot_examples(results["examples"], images, items, preds, codes, class_names, os.path.join(args.fig_dir, "task1_cue_conflict_examples.png"))
    if not args.skip_tsne:
        plot_tsne(clean_out["features"], out["features"], shape_labels, class_names, cfg, os.path.join(args.fig_dir, "task1_tsne_cue_conflict.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="task1/configs/task1.yaml")
    parser.add_argument("--meta", default="task1/results/cue_conflicts_meta.json")
    parser.add_argument("--cache-dir", default="task1/cache")
    parser.add_argument("--out-dir", default="task1/results")
    parser.add_argument("--fig-dir", default="report/figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-tsne", action="store_true")
    main(parser.parse_args())
