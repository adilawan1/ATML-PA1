from __future__ import annotations

"""Common evaluation (Task 4, Step 6). Run after `extract_outputs.py` has cached
{method}_{train,val,test,near,far}.pt for at least Vanilla (and GCSC, once trained).

Produces:
1. `task4/results/table1_vanilla_posthoc_scores.json` -- MSP/MLS/Energy/Mahalanobis on the
   frozen Vanilla model: near/far/all AUROC + validation-calibrated (95th percentile)
   rejection.
2. `task4/results/table2_model_comparison_mls.json` -- Vanilla/GCSC[/PROSER] compared with
   MLS as the common score (CSA + near/far AUROC + rejection). Only includes methods whose
   cache files already exist, so it runs fine before PROSER is implemented.
3. `report/figures/task4_score_distributions.png` -- MSP/MLS/Mahalanobis score histograms
   (known-test vs. near vs. far), the required compact multi-panel figure.
4. `task4/results/failure_cases.json` -- incorrectly-accepted near/far examples under the
   Vanilla MLS threshold (unknown class, predicted class, score, threshold).
5. `task4/results/unknown_class_breakdown.json` -- per unknown class: acceptance rate under each method's
   MLS threshold and which CIFAR-10 labels absorbed the accepted images.

PROSER (once its cache exists) contributes two rows: MLS over its ten KNOWN-class logits (directly
comparable with Vanilla/GCSC; CSA also uses only those logits) and a second row using its
placeholder-based detection score (reference implementation: temperature-1024 softmax over
[known logits, max dummy logit], unknownness = P(dummy) - max P(known)); both use the 95th-percentile
threshold calibrated on CIFAR-10 validation scores only.
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import CIFAR10

from common.metrics import accuracy
from common.plotting import apply_style, save_figure
from task4.data.cifar100_unknowns import CIFAR100UnknownSubset
from task4.evaluation.failure_analysis import find_incorrect_acceptances
from task4.evaluation.thresholds import evaluate_score
from task4.methods.proser import proser_placeholder_score
from task4.scores.energy import energy_score
from task4.scores.mahalanobis import fit_class_gaussians, mahalanobis_score
from task4.scores.mls import mls_score
from task4.scores.msp import msp_score


def load_cache(cache_dir: str, method_name: str, split_name: str) -> Dict[str, torch.Tensor]:
    return torch.load(os.path.join(cache_dir, f"{method_name}_{split_name}.pt"))


def build_table1(cache_dir: str, method_name: str = "vanilla") -> List[Dict]:
    train = load_cache(cache_dir, method_name, "train")
    val = load_cache(cache_dir, method_name, "val")
    test = load_cache(cache_dir, method_name, "test")
    near = load_cache(cache_dir, method_name, "near")
    far = load_cache(cache_dir, method_name, "far")

    means, diag_var = fit_class_gaussians(train["features"].numpy(), train["labels"].numpy())

    score_fns = {
        "MSP": lambda c: msp_score(c["logits"].numpy()),
        "MLS": lambda c: mls_score(c["logits"].numpy()),
        "Energy": lambda c: energy_score(c["logits"].numpy()),
        "Mahalanobis": lambda c: mahalanobis_score(c["features"].numpy(), means, diag_var),
    }

    rows = []
    for name, fn in score_fns.items():
        row = {
            "score": name,
            **evaluate_score(fn(val), fn(test), fn(near), fn(far)),
        }
        rows.append(row)
    return rows


def build_table2(cache_dir: str, methods: Sequence[str]) -> List[Dict]:
    rows = []
    for method in methods:
        val = load_cache(cache_dir, method, "val")
        test = load_cache(cache_dir, method, "test")
        near = load_cache(cache_dir, method, "near")
        far = load_cache(cache_dir, method, "far")

        csa = accuracy(test["labels"].numpy(), test["logits"].argmax(dim=1).numpy())
        row = {
            "method": method,
            "score": "MLS",
            "csa": csa,
            **evaluate_score(
                mls_score(val["logits"].numpy()),
                mls_score(test["logits"].numpy()),
                mls_score(near["logits"].numpy()),
                mls_score(far["logits"].numpy()),
            ),
        }
        rows.append(row)

        if method == "proser":
            placeholder = lambda c: proser_placeholder_score(c["logits"].numpy(), c["dummy_logits"].numpy())  # noqa: E731
            rows.append(
                {
                    "method": method,
                    "score": "placeholder",
                    "csa": csa,
                    **evaluate_score(placeholder(val), placeholder(test), placeholder(near), placeholder(far)),
                }
            )
    return rows


def plot_score_distributions(cache_dir: str, method_name: str, save_path: str) -> None:
    train = load_cache(cache_dir, method_name, "train")
    test = load_cache(cache_dir, method_name, "test")
    near = load_cache(cache_dir, method_name, "near")
    far = load_cache(cache_dir, method_name, "far")
    means, diag_var = fit_class_gaussians(train["features"].numpy(), train["labels"].numpy())

    panels = {
        "MSP": (msp_score(test["logits"].numpy()), msp_score(near["logits"].numpy()), msp_score(far["logits"].numpy())),
        "MLS": (mls_score(test["logits"].numpy()), mls_score(near["logits"].numpy()), mls_score(far["logits"].numpy())),
        "Mahalanobis": (
            mahalanobis_score(test["features"].numpy(), means, diag_var),
            mahalanobis_score(near["features"].numpy(), means, diag_var),
            mahalanobis_score(far["features"].numpy(), means, diag_var),
        ),
    }

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, (known, near_scores, far_scores)) in zip(axes, panels.items()):
        ax.hist(known, bins=40, alpha=0.5, density=True, label="known (test)")
        ax.hist(near_scores, bins=40, alpha=0.5, density=True, label="near unknown")
        ax.hist(far_scores, bins=40, alpha=0.5, density=True, label="far unknown")
        ax.set_title(f"{method_name}: {name}")
        ax.legend(fontsize=8)
    save_figure(fig, save_path)
    print(f"saved {save_path}")


def build_failure_cases(cache_dir: str, data_root: str, table1_rows: List[Dict], k: int = 3) -> Dict[str, List[Dict]]:
    threshold = next(r for r in table1_rows if r["score"] == "MLS")["threshold_tau"]
    cifar10_classes = CIFAR10(root=data_root, train=False, download=True).classes

    result = {}
    for group in ("near", "far"):
        cache = load_cache(cache_dir, "vanilla", group)
        dataset = CIFAR100UnknownSubset(data_root, group=group)
        scores = mls_score(cache["logits"].numpy())
        predicted_names = [cifar10_classes[p] for p in cache["logits"].argmax(dim=1).numpy()]
        unknown_names = [dataset.class_names[label] for label in dataset.original_labels]

        cases = find_incorrect_acceptances(scores, threshold, unknown_names, predicted_names)
        result[group] = cases[:k] if len(cases) > k else cases
        print(f"{group}: {len(cases)} incorrectly accepted (showing up to {k})")
    return result


def unknown_breakdown(scores: np.ndarray, predicted: Sequence[str], unknown_names: Sequence[str], threshold: float) -> Dict[str, Dict]:
    """Per unknown fine class: how many test images were ACCEPTED as known under `threshold`
    (accept when score <= threshold), and which CIFAR-10 labels absorbed them (Research Question 1)."""
    names = np.asarray(unknown_names)
    predicted = np.asarray(predicted)
    out = {}
    for cls in sorted(set(names.tolist())):
        in_class = names == cls
        accepted = in_class & (scores <= threshold)
        out[cls] = {
            "n": int(in_class.sum()),
            "accepted": int(accepted.sum()),
            "accept_rate": float(accepted.sum() / in_class.sum()),
            "absorbed_by": dict(Counter(predicted[accepted].tolist()).most_common()),
        }
    return out


def build_unknown_breakdown(cache_dir: str, data_root: str, methods: Sequence[str], table2_rows: List[Dict]) -> Dict:
    """Per-method (MLS score, that method's own validation-calibrated threshold) breakdown for the near and far groups."""
    cifar10_classes = CIFAR10(root=data_root, train=False, download=True).classes
    result = {}
    for method in methods:
        tau = next(r for r in table2_rows if r["method"] == method and r["score"] == "MLS")["threshold_tau"]
        result[method] = {"threshold_tau": tau}
        for group in ("near", "far"):
            cache = load_cache(cache_dir, method, group)
            dataset = CIFAR100UnknownSubset(data_root, group=group)
            predicted = [cifar10_classes[p] for p in cache["logits"].argmax(dim=1).numpy()]
            names = [dataset.class_names[label] for label in dataset.original_labels]
            result[method][group] = unknown_breakdown(mls_score(cache["logits"].numpy()), predicted, names, tau)
    return result


def main(data_root: str, cache_dir: str) -> None:
    os.makedirs("task4/results", exist_ok=True)

    table1 = build_table1(cache_dir, "vanilla")
    with open("task4/results/table1_vanilla_posthoc_scores.json", "w") as f:
        json.dump(table1, f, indent=2)
    print("\nTable 1 -- Vanilla, post-hoc scores:")
    for row in table1:
        print(row)

    available_methods = [m for m in ("vanilla", "gcsc", "proser") if os.path.exists(os.path.join(cache_dir, f"{m}_test.pt"))]
    table2 = build_table2(cache_dir, available_methods)
    with open("task4/results/table2_model_comparison_mls.json", "w") as f:
        json.dump(table2, f, indent=2)
    print("\nTable 2 -- model comparison (MLS):")
    for row in table2:
        print(row)
    if "proser" not in available_methods:
        print("\n(PROSER not yet cached -- table 2 will regenerate with its row, plus a separate\n"
              " placeholder-score row, once task4/methods/proser.py is implemented and cached.)")

    with open("task4/results/unknown_class_breakdown.json", "w") as f:
        json.dump(build_unknown_breakdown(cache_dir, data_root, available_methods, table2), f, indent=2)

    plot_score_distributions(cache_dir, "vanilla", "report/figures/task4_score_distributions.png")

    failures = build_failure_cases(cache_dir, data_root, table1)
    with open("task4/results/failure_cases.json", "w") as f:
        json.dump(failures, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", default="task4/cache")
    args = parser.parse_args()
    main(args.data_root, args.cache_dir)
