from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from torchvision.datasets import STL10

SUBSET_SEED = 6304
SUBSET_SIZE = 500
DEFAULT_OUT_PATH = "task1/data/eval_subset_seed6304.json"


def build_eval_subset(data_root: str, out_path: str = DEFAULT_OUT_PATH, subset_size: int = SUBSET_SIZE) -> Dict:
    """Class-balanced subset of `subset_size` official STL-10 TEST images (seed 6304).

    STL-10's test split has 800 images/class (8000 total, 10 classes), so an even split is
    `subset_size // num_classes` per class with no imbalance to document. If a different
    dataset/class ever has fewer available images than the even split, fall back to using
    all of them and record the shortfall in `imbalance_notes`.
    """
    test_set = STL10(root=data_root, split="test", download=True)
    labels = np.array(test_set.labels)
    classes = sorted(set(labels.tolist()))
    per_class = subset_size // len(classes)

    rng = np.random.RandomState(SUBSET_SEED)
    selected_indices: List[int] = []
    imbalance_notes: Dict[str, str] = {}

    for c in classes:
        class_indices = np.flatnonzero(labels == c)
        if len(class_indices) < per_class:
            imbalance_notes[str(c)] = f"only {len(class_indices)} available, needed {per_class}"
            chosen = class_indices
        else:
            chosen = rng.choice(class_indices, size=per_class, replace=False)
        selected_indices.extend(int(i) for i in chosen)

    selected_indices.sort()
    subset = {
        "seed": SUBSET_SEED,
        "split": "test",
        "dataset": "STL-10",
        "indices": selected_indices,
        "labels": [int(labels[i]) for i in selected_indices],
        "imbalance_notes": imbalance_notes,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(subset, f, indent=2)
    return subset


def load_eval_subset(path: str = DEFAULT_OUT_PATH) -> Dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args()
    subset = build_eval_subset(args.data_root, args.out)
    print(f"Wrote {len(subset['indices'])}-image eval subset to {args.out}")
    if subset["imbalance_notes"]:
        print(f"Imbalance: {subset['imbalance_notes']}")
