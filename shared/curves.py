from __future__ import annotations

"""Training-curve figure for Tasks 2/3, read straight from the per-epoch `metrics.jsonl` logs so the
plot can be regenerated without retraining. One panel per quantity that any run logged: classification
loss, alignment penalty (MMD), domain loss and domain-discriminator accuracy (DANN/CDAN; accuracy near
0.5 can mean confusion *or* an undertrained discriminator -- read it next to the other panels), and the
mean source-validation macro-F1 used for checkpoint selection."""

from typing import Dict, List

import matplotlib.pyplot as plt

from common.logging import read_metrics
from common.plotting import apply_style, save_figure

PANELS = [
    ("cls_loss", "classification loss"),
    ("mmd", "MMD^2 penalty"),
    ("domain_loss", "domain loss"),
    ("domain_acc", "domain discriminator acc."),
    ("mean_source_val_macro_f1", "mean source-val macro-F1"),
]


def plot_curves(runs: Dict[str, str], path: str) -> None:
    """`runs`: {label: path to metrics.jsonl}."""
    logs = {label: read_metrics(p) for label, p in runs.items()}
    panels: List = [(key, title) for key, title in PANELS if any(key in row for rows in logs.values() for row in rows)]
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.3), squeeze=False)
    for ax, (key, title) in zip(axes[0], panels):
        for label, rows in logs.items():
            points = [(r["epoch"], r[key]) for r in rows if key in r]
            if points:
                ax.plot(*zip(*points), marker=".", label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, path)


def plot_confusions(results: Dict[str, Dict], class_names: List[str], path: str) -> None:
    """Row-normalized target confusion matrices, one panel per run, for the per-class failure analysis
    (`results[run]["target"]["confusion"]`; rows = true class, columns = predicted class)."""
    import numpy as np

    apply_style()
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        conf = np.array(r["target"]["confusion"], dtype=float)
        norm = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"{name}  (acc {r['target']['accuracy']:.3f})", fontsize=8)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("predicted", fontsize=7)
        ax.grid(False)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if norm[i, j] >= 0.1:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white" if norm[i, j] > 0.5 else "black")
    axes[0][0].set_ylabel("true", fontsize=7)
    fig.tight_layout()
    save_figure(fig, path)


def plot_study(table, xlabel: str, columns: Dict[str, str], path: str, log_x: bool = False) -> None:
    """Compact plot for a controlled alignment/regularization-strength study: one panel per metric vs. strength.
    `table` is a DataFrame with a 'strength' column; `columns` maps column name -> panel title."""
    apply_style()
    fig, axes = plt.subplots(1, len(columns), figsize=(3.4 * len(columns), 3.2), squeeze=False)
    for ax, (col, title) in zip(axes[0], columns.items()):
        ax.plot(table["strength"], table[col], marker="o")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_xscale("log" if log_x else "linear")
    fig.tight_layout()
    save_figure(fig, path)
