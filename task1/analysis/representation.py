from __future__ import annotations

from typing import List

import numpy as np
from sklearn.manifold import TSNE

SEED = 6304


def fit_tsne(features: np.ndarray, perplexity: float = 30.0, seed: int = SEED) -> np.ndarray:
    """Fit ONE 2D t-SNE projection to `features` (clean + transformed, concatenated) so both
    conditions land in the same space. Per the assignment, never compare coordinates across
    projections fit separately for different backbones/conditions.
    """
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(features)


def fit_umap(features: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = SEED) -> np.ndarray:
    import umap  # optional dependency (umap-learn); only imported if UMAP is the chosen method

    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed).fit_transform(features)


def draw_projection(ax, coords: np.ndarray, class_labels: np.ndarray, is_transformed: np.ndarray, title: str) -> None:
    """Color = ground-truth class, marker = clean ('o') vs. transformed ('x') -- Step 6's figure."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    for transformed, marker in [(False, "o"), (True, "x")]:
        mask = is_transformed == transformed
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[cmap(int(c) % 10) for c in class_labels[mask]],
            marker=marker,
            s=10,
            alpha=0.7,
            linewidths=0.8,
        )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def legend_handles(class_names: List[str]):
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=cmap(i % 10), label=name, markersize=6)
        for i, name in enumerate(class_names)
    ]
    handles += [
        Line2D([0], [0], marker="o", linestyle="", color="gray", label="clean", markersize=6),
        Line2D([0], [0], marker="x", linestyle="", color="gray", label="transformed", markersize=6),
    ]
    return handles
