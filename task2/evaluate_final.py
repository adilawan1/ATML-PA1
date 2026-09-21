from __future__ import annotations

"""Task 2 common evaluation + alignment diagnostic (Step 5). Run only after every checkpoint and
setting is fixed: this is the first place Sketch LABELS are read.

Per run: accuracy / macro-F1 on each source-validation domain (and their mean), target accuracy /
macro-F1, target-accuracy change vs. Source-only, domain separability (frozen backbone, equal numbers
of source-validation and target features, 70/30 split with seed 6304, balanced logistic regression
C=1; 50% = chance), per-class target accuracy, and -- for the classes with the largest gain/loss vs.
Source-only -- their dominant confusions, so a class-specific negative transfer hidden behind an
aggregate gain is visible.
"""

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from common.metrics import accuracy, macro_f1, per_class_accuracy
from shared.eval_utils import predict, source_val_metrics
from shared.pacs import CLASSES
from shared.pacs_data import SOURCE_DOMAINS, TARGET_DOMAIN, PACSData
from task2.evaluation.class_analysis import biggest_class_shifts, dominant_confusion
from task2.evaluation.domain_separability import domain_separability_score
from task2.models.backbone import build_resnet18_backbone

SEED = 6304


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    model = build_resnet18_backbone(num_classes=7).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    return model.eval()


def evaluate_run(model, data: PACSData, device: str) -> Dict:
    target_rows = data.rows(TARGET_DOMAIN, "all")
    target_labels = data.target_labels_for_final_eval().numpy()  # first use of target labels
    target_feats, target_logits = predict(model, data, target_rows, device)
    preds = target_logits.argmax(1).numpy()

    source_feats = torch.cat([predict(model, data, data.rows(d, "val"), device)[0] for d in SOURCE_DOMAINS])
    chosen = np.random.RandomState(SEED).choice(len(target_feats), size=len(source_feats), replace=False)
    separability = domain_separability_score(source_feats.numpy(), target_feats[chosen].numpy())

    return {
        "source_val": source_val_metrics(model, data, device),
        "target": {
            "accuracy": accuracy(target_labels, preds),
            "macro_f1": macro_f1(target_labels, preds),
            "per_class_accuracy": [float(v) for v in per_class_accuracy(target_labels, preds, len(CLASSES))],
            "confusion": confusion_matrix(target_labels, preds, labels=list(range(len(CLASSES)))).tolist(),
        },
        "domain_separability": separability,
    }


def summary_table(results: Dict[str, Dict], baseline: str = "source_only") -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        row = {}
        for d in SOURCE_DOMAINS:
            row[f"{d}_acc"] = r["source_val"]["per_domain"][d]["accuracy"]
            row[f"{d}_f1"] = r["source_val"]["per_domain"][d]["macro_f1"]
        row["mean_src_acc"] = r["source_val"]["mean_accuracy"]
        row["mean_src_f1"] = r["source_val"]["mean_macro_f1"]
        row["target_acc"] = r["target"]["accuracy"]
        row["target_f1"] = r["target"]["macro_f1"]
        row["target_acc_change"] = r["target"]["accuracy"] - results[baseline]["target"]["accuracy"]
        row["domain_separability"] = r["domain_separability"]
        rows[name] = row
    return pd.DataFrame(rows).T


def class_analysis(results: Dict[str, Dict], baseline: str = "source_only", k: int = 2) -> Dict:
    base = results[baseline]["target"]
    base_conf = np.array(base["confusion"])
    out = {}
    for name, r in results.items():
        if name == baseline:
            continue
        conf = np.array(r["target"]["confusion"])
        improved, degraded = biggest_class_shifts(np.array(base["per_class_accuracy"]), np.array(r["target"]["per_class_accuracy"]), k)
        entry = {"per_class_change": {CLASSES[c]: r["target"]["per_class_accuracy"][c] - base["per_class_accuracy"][c] for c in range(len(CLASSES))}}
        for label, classes in (("most_improved", improved), ("most_degraded", degraded)):
            entry[label] = []
            for c in classes:
                wrong_base, share_base = dominant_confusion(base_conf, int(c))
                wrong_run, share_run = dominant_confusion(conf, int(c))
                entry[label].append(
                    {
                        "class": CLASSES[int(c)],
                        "acc_baseline": base["per_class_accuracy"][int(c)],
                        "acc_run": r["target"]["per_class_accuracy"][int(c)],
                        "dominant_confusion_baseline": {"as": CLASSES[wrong_base], "share": share_base},
                        "dominant_confusion_run": {"as": CLASSES[wrong_run], "share": share_run},
                    }
                )
        out[name] = entry
    return out


def evaluate_all(run_names: List[str], data: PACSData, checkpoint_for, results_dir: str, device: str, main_runs: List[str], verbose: bool = True) -> Dict:
    results = {}
    for name in run_names:
        results[name] = evaluate_run(load_model(checkpoint_for(name), device), data, device)
        if verbose:
            print(f"{name:16s} target acc {results[name]['target']['accuracy']:.4f} | domain separability {results[name]['domain_separability']:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    table = summary_table(results)
    table.to_csv(os.path.join(results_dir, "summary.csv"))
    main_present = [r for r in main_runs if r in results]
    if "source_only" in results:
        with open(os.path.join(results_dir, "class_analysis.json"), "w") as f:
            json.dump(class_analysis({r: results[r] for r in main_present}), f, indent=2)
    if verbose:
        print("\n" + table.round(4).to_string())
    else:
        print(f"evaluation written to {results_dir}/summary.csv (blind mode: not printed)")
    return results
