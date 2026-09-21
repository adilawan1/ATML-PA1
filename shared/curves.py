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
