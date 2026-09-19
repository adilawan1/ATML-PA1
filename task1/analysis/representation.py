from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.manifold import TSNE

SEED = 6304


def fit_tsne(features: np.ndarray, perplexity: float = 30.0, seed: int = SEED) -> np.ndarray:
    """Fit one 2D t-SNE projection to `features` (clean + transformed, concatenated) so both
    conditions land in the same space -- per the assignment, never compare coordinates across
    projections fit separately for different backbones.
    """
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(features)


def fit_umap(features: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = SEED) -> np.ndarray:
    import umap  # optional dependency (umap-learn); only imported if UMAP is the chosen method

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(features)


def plot_projection(
    coords: np.ndarray,
    class_labels: np.ndarray,
    is_transformed: np.ndarray,
    class_names: list,
    title: str,
    save_path: Optional[str] = None,
):
    """Color = ground-truth class, marker = clean vs. transformed (Step 6's required figure)."""
    import matplotlib.pyplot as plt

    from common.plotting import apply_style, save_figure

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    for c in np.unique(class_labels):
        for transformed, marker in [(False, "o"), (True, "x")]:
            mask = (class_labels == c) & (is_transformed == transformed)
            if not mask.any():
                continue
            label = f"{class_names[c]} ({'transformed' if transformed else 'clean'})"
            ax.scatter(coords[mask, 0], coords[mask, 1], color=cmap(c % 10), marker=marker, s=12, alpha=0.7, label=label)
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2, loc="best")
    if save_path:
        save_figure(fig, save_path)
    return fig
