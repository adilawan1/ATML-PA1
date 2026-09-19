from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

STYLE = {
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save_figure(fig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
