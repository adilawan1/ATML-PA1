from __future__ import annotations

"""Shape/texture cue-conflict generation (Task 1, Step 3). NO classifier is imported or run here:
the accepted set is fixed by the rule below, from image statistics alone, before any model is
evaluated -- the assignment forbids using model predictions to decide what to keep.

For every unordered class pair (A, B) and both directions, AdaIN stylizes a content image of the
shape class with a style image of the texture class:  shape/content = A, texture/style = B  and
shape/content = B, texture/style = A.  Content images come from the fixed 500-image evaluation
subset (so each conflict has a clean counterpart whose features are already evaluated); style
images come from the remaining STL-10 test images of the texture class.

Pre-registered rejection rule (thresholds live in `configs/task1.yaml`; calibrate them on the
`--preview` contact sheets -- looking at images only -- then leave them fixed). A stylization is
REJECTED unless all three hold:
  1. structure preserved: correlation between coarse (8x8-pooled) Sobel gradient-magnitude maps of
     the content image and the stylized image >= min_structure_corr  (the object's silhouette/edge
     layout survived, so the shape cue is still present);
  2. style applied: the stylized image's per-channel color mean/std moved toward the style image's by
     at least min_style_shift, where shift = 1 - d(stylized, style) / d(content, style)
     (the texture/appearance cue actually changed);
  3. a real conflict existed: d(content, style) >= min_style_gap (if the two images already had
     near-identical color statistics there was nothing to transfer).
These are coarse image-statistic proxies for "shape still visible, style visibly applied"; the
accepted/rejected contact sheets are saved so the rule can be inspected by eye.

Balancing: up to `keep_per_cell` accepted items are kept per (pair, direction) cell, so the final
set is as balanced across pairs and directions as the accepted counts allow.
"""

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml

from task1.data.adain import DEFAULT_REPO_DIR, AdaINStylizer
from task1.data.make_subset import load_eval_subset
from task1.data.stl10 import load_common_tensors, load_stl10

RULE_DESCRIPTION = (
    "Reject unless (1) coarse Sobel gradient-magnitude correlation(content, stylized) >= min_structure_corr, "
    "(2) color-statistics shift toward the style image >= min_style_shift, and "
    "(3) color-statistics gap(content, style) >= min_style_gap. Image statistics only; no model predictions."
)


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    gray = TF.rgb_to_grayscale(x)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, kx.transpose(2, 3), padding=1)
    return (gx**2 + gy**2).sqrt()


def structure_correlation(content: torch.Tensor, stylized: torch.Tensor, pool: int = 8) -> torch.Tensor:
    a = F.avg_pool2d(_gradient_magnitude(content), pool).flatten(1)
    b = F.avg_pool2d(_gradient_magnitude(stylized), pool).flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


def _color_stats(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(2)
    return torch.cat([flat.mean(2), flat.std(2)], dim=1)


def style_gap(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    return (_color_stats(content) - _color_stats(style)).norm(dim=1)


def style_shift(content: torch.Tensor, style: torch.Tensor, stylized: torch.Tensor) -> torch.Tensor:
    gap = style_gap(content, style)
    return 1.0 - (_color_stats(stylized) - _color_stats(style)).norm(dim=1) / (gap + 1e-8)


def image_metrics(content, style, stylized) -> Dict[str, torch.Tensor]:
    return {
        "structure_corr": structure_correlation(content, stylized),
        "style_shift": style_shift(content, style, stylized),
        "style_gap": style_gap(content, style),
    }


def passes_rule(metrics: Dict[str, torch.Tensor], rule: Dict) -> torch.Tensor:
    return (
        (metrics["structure_corr"] >= rule["min_structure_corr"])
        & (metrics["style_shift"] >= rule["min_style_shift"])
        & (metrics["style_gap"] >= rule["min_style_gap"])
    )


def build_cells(pairs: List[List[str]], classes: List[str]) -> List[Dict]:
    cells = []
    for a, b in pairs:
        for shape_name, texture_name in [(a, b), (b, a)]:
            cells.append(
                {
                    "pair": f"{a}-{b}",
                    "direction": f"shape={shape_name}, texture={texture_name}",
                    "shape_name": shape_name,
                    "texture_name": texture_name,
                    "shape_label": classes.index(shape_name),
                    "texture_label": classes.index(texture_name),
                }
            )
    return cells


def sample_cell_ids(cell: Dict, labels: np.ndarray, subset_indices: List[int], n: int, rng: np.random.RandomState):
    """`n` content ids from the eval subset (shape class) and `n` style ids from the remaining
    test images (texture class); drawn with the cell's own seeded RNG."""
    subset = set(subset_indices)
    content_pool = [i for i in subset_indices if labels[i] == cell["shape_label"]]
    style_pool = [i for i in range(len(labels)) if labels[i] == cell["texture_label"] and i not in subset]
    n_content = min(n, len(content_pool))
    return rng.choice(content_pool, size=n_content, replace=False), rng.choice(style_pool, size=n_content, replace=False)


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    return (x * 255.0).round().clamp(0, 255).to(torch.uint8)


def save_triplet_sheet(triplets, captions: List[str], path: str, per_row: int = 2) -> None:
    import matplotlib.pyplot as plt

    from common.plotting import save_figure

    rows = math.ceil(len(triplets) / per_row)
    fig, axes = plt.subplots(rows, per_row * 3, figsize=(per_row * 3 * 1.5, rows * 1.75), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (content, style, conflict) in enumerate(triplets):
        row, col0 = divmod(k, per_row)
        col0 *= 3
        for j, (img, name) in enumerate(zip((content, style, conflict), ("content", "style", "conflict"))):
            ax = axes[row][col0 + j]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(name if row == 0 else "", fontsize=6)
        axes[row][col0].text(0, -0.08, captions[k], transform=axes[row][col0].transAxes, fontsize=5.5, va="top")
    fig.tight_layout()
    save_figure(fig, path)


def generate(cfg: Dict, data_root: str, cache_dir: str, out_meta: str, fig_dir: str, device: str, adain_dir: str) -> Dict:
    cc = cfg["cue_conflicts"]
    test_set = load_stl10(data_root, "test")
    labels = np.asarray(test_set.labels)
    classes = list(test_set.classes)
    subset = load_eval_subset(cfg["eval_subset"]["path"])
    stylizer = AdaINStylizer(device, adain_dir)

    cells = build_cells(cc["pairs"], classes)
    kept_images, kept_items, rejected_items, cell_reports = [], [], [], []
    samples_kept, samples_rejected = [], []

    for cell_idx, cell in enumerate(cells):
        rng = np.random.RandomState(cc["seed"] + cell_idx)
        content_ids, style_ids = sample_cell_ids(cell, labels, subset["indices"], cc["candidates_per_cell"], rng)
        content = load_common_tensors(test_set, content_ids.tolist())
        style = load_common_tensors(test_set, style_ids.tolist())
        stylized = stylizer.stylize(content, style, cc["alpha"])
        metrics = image_metrics(content, style, stylized)
        accepted = passes_rule(metrics, cc["rejection_rule"]).tolist()

        n_kept = 0
        first_kept = first_rejected = None
        for k in range(len(content_ids)):
            item = {
                "pair": cell["pair"],
                "direction": cell["direction"],
                "shape_label": cell["shape_label"],
                "texture_label": cell["texture_label"],
                "content_idx": int(content_ids[k]),
                "style_idx": int(style_ids[k]),
                **{name: float(values[k]) for name, values in metrics.items()},
            }
            if accepted[k] and n_kept < cc["keep_per_cell"]:
                item["position"] = len(kept_images)
                kept_images.append(to_uint8(stylized[k]))
                kept_items.append(item)
                n_kept += 1
                first_kept = first_kept if first_kept is not None else k
            elif not accepted[k]:
                rejected_items.append(item)
                first_rejected = first_rejected if first_rejected is not None else k

        caption = f"{cell['shape_name']} shape / {cell['texture_name']} texture"
        if first_kept is not None:
            samples_kept.append(((content[first_kept], style[first_kept], stylized[first_kept]), caption))
        if first_rejected is not None:
            samples_rejected.append(((content[first_rejected], style[first_rejected], stylized[first_rejected]), caption))
        cell_reports.append(
            {
                "pair": cell["pair"],
                "direction": cell["direction"],
                "candidates": len(content_ids),
                "accepted_by_rule": int(sum(accepted)),
                "rejected_by_rule": len(content_ids) - int(sum(accepted)),
                "kept": n_kept,
            }
        )
        print(f"{cell['pair']:>14s} | {cell['direction']:<32s} candidates={len(content_ids)} accepted={sum(accepted)} kept={n_kept}")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"images": torch.stack(kept_images)}, os.path.join(cache_dir, "cue_conflicts.pt"))

    meta = {
        "rule_description": RULE_DESCRIPTION,
        "config": cc,
        "cells": cell_reports,
        "totals": {
            "candidates": sum(c["candidates"] for c in cell_reports),
            "accepted_by_rule": sum(c["accepted_by_rule"] for c in cell_reports),
            "rejected_by_rule": sum(c["rejected_by_rule"] for c in cell_reports),
            "kept": len(kept_items),
        },
        "kept": kept_items,
        "rejected": rejected_items,
    }
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=1)

    os.makedirs(fig_dir, exist_ok=True)
    save_triplet_sheet([t for t, _ in samples_kept], [c for _, c in samples_kept], os.path.join(fig_dir, "task1_cue_conflict_accepted.png"))
    if samples_rejected:
        save_triplet_sheet([t for t, _ in samples_rejected], [c for _, c in samples_rejected], os.path.join(fig_dir, "task1_cue_conflict_rejected.png"))
    print(f"\nkept {len(kept_items)} conflicts (need >= 200); rejected {meta['totals']['rejected_by_rule']} of {meta['totals']['candidates']} candidates")
    return meta


def preview(cfg: Dict, data_root: str, fig_dir: str, device: str, adain_dir: str, per_cell: int = 20) -> None:
    """Calibration aid (images and image statistics only): stylize a few candidates per cell at
    each `preview_alphas` strength, print the statistics + pass rate under the current rule, and
    save a contact sheet so alpha and the rule thresholds can be chosen by eye."""
    import matplotlib.pyplot as plt

    from common.plotting import save_figure

    cc = cfg["cue_conflicts"]
    test_set = load_stl10(data_root, "test")
    labels = np.asarray(test_set.labels)
    subset = load_eval_subset(cfg["eval_subset"]["path"])
    stylizer = AdaINStylizer(device, adain_dir)
    cells = build_cells(cc["pairs"], list(test_set.classes))
    alphas = cc["preview_alphas"]

    stats = {a: {"structure_corr": [], "style_shift": [], "style_gap": [], "pass": []} for a in alphas}
    sheet_rows = []
    for cell_idx, cell in enumerate(cells):
        rng = np.random.RandomState(cc["seed"] + cell_idx)
        content_ids, style_ids = sample_cell_ids(cell, labels, subset["indices"], per_cell, rng)
        content = load_common_tensors(test_set, content_ids.tolist())
        style = load_common_tensors(test_set, style_ids.tolist())
        row = {"caption": f"{cell['shape_name']} shape / {cell['texture_name']} texture", "content": content[0], "style": style[0], "outs": {}}
        for alpha in alphas:
            stylized = stylizer.stylize(content, style, alpha)
            m = image_metrics(content, style, stylized)
            ok = passes_rule(m, cc["rejection_rule"])
            for name in ("structure_corr", "style_shift", "style_gap"):
                stats[alpha][name].append(m[name])
            stats[alpha]["pass"].append(ok.float())
            row["outs"][alpha] = (stylized[0], float(m["structure_corr"][0]), float(m["style_shift"][0]))
        sheet_rows.append(row)

    print(f"\nrule: {cc['rejection_rule']}")
    print("alpha | median structure_corr | median style_shift | median style_gap | pass rate")
    for alpha in alphas:
        cat = {k: torch.cat(v) for k, v in stats[alpha].items()}
        print(
            f"{alpha:5.2f} | {cat['structure_corr'].median():21.3f} | {cat['style_shift'].median():18.3f} | "
            f"{cat['style_gap'].median():16.3f} | {cat['pass'].mean():.2f}"
        )

    n_cols = 2 + len(alphas)
    fig, axes = plt.subplots(len(sheet_rows), n_cols, figsize=(1.6 * n_cols, 1.75 * len(sheet_rows)), squeeze=False)
    for r, row in enumerate(sheet_rows):
        images = [row["content"], row["style"]] + [row["outs"][a][0] for a in alphas]
        titles = ["content", "style"] + [f"a={a}  c={row['outs'][a][1]:.2f} s={row['outs'][a][2]:.2f}" for a in alphas]
        for c, (img, title) in enumerate(zip(images, titles)):
            axes[r][c].imshow(img.permute(1, 2, 0).numpy())
            axes[r][c].axis("off")
            axes[r][c].set_title(title if r == 0 else "", fontsize=5.5)
        axes[r][0].text(-0.05, 0.5, row["caption"], transform=axes[r][0].transAxes, fontsize=5.5, ha="right", va="center")
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    save_figure(fig, os.path.join(fig_dir, "task1_cue_conflict_preview.png"))
    print(f"saved {os.path.join(fig_dir, 'task1_cue_conflict_preview.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="task1/configs/task1.yaml")
    parser.add_argument("--cache-dir", default="task1/cache")
    parser.add_argument("--meta-out", default="task1/results/cue_conflicts_meta.json")
    parser.add_argument("--fig-dir", default="report/figures")
    parser.add_argument("--adain-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preview", action="store_true", help="calibration contact sheet only; writes no dataset")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.preview:
        preview(config, args.data_root, args.fig_dir, args.device, args.adain_dir)
    else:
        generate(config, args.data_root, args.cache_dir, args.meta_out, args.fig_dir, args.device, args.adain_dir)
