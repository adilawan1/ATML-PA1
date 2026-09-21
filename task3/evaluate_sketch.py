from __future__ import annotations

"""Task 3 final evaluation -- the ONLY place Sketch is opened, run after `task3/run_experiments.py` has
finished and every Task 3 model / setting is frozen.

For ERM (= the Task 2 Source-only checkpoint), DAN-DG and SAM (plus the controlled-study runs): source
validation accuracy / macro-F1 per domain with mean and worst-domain values, Sketch accuracy / macro-F1 and
the change vs. ERM, source-domain separability and the local-sharpness proxy (both from the Sketch-free
diagnostics file), per-class Sketch accuracy changes vs. ERM with dominant confusions, and a comparison
with the Task 2 results (target-aware DAN/DANN/CDAN vs. target-free DAN-DG/SAM, same ERM baseline).
"""

import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from common.metrics import accuracy, macro_f1, per_class_accuracy
from shared.curves import plot_confusions, plot_study
from shared.eval_utils import predict
from shared.pacs import CLASSES
from shared.pacs_data import TARGET_DOMAIN, PACSData
from shared.pacs_protocol import load_pacs_protocol
from task2.evaluate_final import class_analysis, load_model
from task3.run_experiments import MAIN_RUNS, RESULTS_DIR, STUDY_RUNS, STUDY_VALUES, checkpoint_for


def sketch_metrics(model, data: PACSData, device: str) -> dict:
    labels = data.target_labels_for_final_eval().numpy()  # first use of Sketch labels in Task 3
    _, logits = predict(model, data, data.rows(TARGET_DOMAIN, "all"), device)
    preds = logits.argmax(1).numpy()
    return {
        "accuracy": accuracy(labels, preds),
        "macro_f1": macro_f1(labels, preds),
        "per_class_accuracy": [float(v) for v in per_class_accuracy(labels, preds, len(CLASSES))],
        "confusion": confusion_matrix(labels, preds, labels=list(range(len(CLASSES)))).tolist(),
    }


def main(args) -> None:
    device = args.device
    with open(os.path.join(RESULTS_DIR, "source_diagnostics.json")) as f:
        diagnostics = json.load(f)
    names = list(diagnostics)  # erm first, then every trained run
    data = PACSData(load_pacs_protocol(args.protocol), args.pacs_root, cache_path=args.cache, include_target=True)

    sketch = {name: sketch_metrics(load_model(checkpoint_for(args.ckpt_root, name), device), data, device) for name in names}
    erm_acc = sketch["erm"]["accuracy"]

    rows = {}
    for name in names:
        sv = diagnostics[name]["source_val"]
        row = {f"{d}_acc": sv["per_domain"][d]["accuracy"] for d in sv["per_domain"]}
        row.update({f"{d}_f1": sv["per_domain"][d]["macro_f1"] for d in sv["per_domain"]})
        row.update(
            mean_src_acc=sv["mean_accuracy"], mean_src_f1=sv["mean_macro_f1"], worst_src_acc=sv["worst_accuracy"], worst_src_f1=sv["worst_macro_f1"],
            sketch_acc=sketch[name]["accuracy"], sketch_f1=sketch[name]["macro_f1"], sketch_acc_change_vs_erm=sketch[name]["accuracy"] - erm_acc,
            source_domain_separability=diagnostics[name]["source_domain_separability"], sharpness=diagnostics[name]["sharpness"],
        )
        rows[name] = row
    table = pd.DataFrame(rows).T
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    blind = getattr(args, "blind", False)
    if not blind:
        print(table.round(4).to_string())

    with open(os.path.join(RESULTS_DIR, "sketch_results.json"), "w") as f:
        json.dump(sketch, f, indent=2)
    main_present = [r for r in ["erm"] + MAIN_RUNS if r in sketch]
    with open(os.path.join(RESULTS_DIR, "class_analysis.json"), "w") as f:
        json.dump(class_analysis({r: {"target": sketch[r]} for r in main_present}, baseline="erm"), f, indent=2)

    # Task 2 vs Task 3 on the same Sketch target, same ERM baseline
    task2_path = "task2/results/summary.json"
    if os.path.exists(task2_path):
        with open(task2_path) as f:
            task2 = json.load(f)
        comparison = {"erm (source-only)": sketch["erm"]}
        comparison.update({f"task2/{k}": v["target"] for k, v in task2.items() if k in ("dan", "dann", "cdan")})
        comparison.update({f"task3/{k}": sketch[k] for k in MAIN_RUNS if k in sketch})
        cmp_table = pd.DataFrame(
            {
                name: {"sketch_acc": m["accuracy"], "sketch_f1": m["macro_f1"], "change_vs_erm": m["accuracy"] - erm_acc,
                       **{f"acc_{c}": m["per_class_accuracy"][i] for i, c in enumerate(CLASSES)}}
                for name, m in comparison.items()
            }
        ).T
        cmp_table.to_csv(os.path.join(RESULTS_DIR, "task2_vs_task3.csv"))
        if not blind:
            print("\nTask 2 (target-aware) vs Task 3 (target-free), same target and ERM baseline:\n" + cmp_table.round(4).to_string())
    else:
        print("(task2/results/summary.json not found -- skipping the Task 2 vs Task 3 comparison)")

    fig_dir = getattr(args, "fig_dir", "report/figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_confusions({r: {"target": sketch[r]} for r in main_present}, CLASSES, os.path.join(fig_dir, "task3_confusions.png"))

    for study, names_in_study in STUDY_RUNS.items():  # the chosen study is the one with a non-main member trained
        if any(n in rows for n in names_in_study if n not in MAIN_RUNS):
            label, values, log_x = STUDY_VALUES[study]
            study_rows = [{"run": n, "strength": v, **{k: rows[n][k] for k in ("mean_src_f1", "source_domain_separability", "sharpness", "sketch_acc")}}
                          for n, v in zip(names_in_study, values) if n in rows]
            study_table = pd.DataFrame(study_rows)
            study_table.to_csv(os.path.join(RESULTS_DIR, "study.csv"), index=False)
            plot_study(study_table, label, {"mean_src_f1": "mean source-val macro-F1", "source_domain_separability": "source-domain separability",
                                            "sharpness": "sharpness proxy", "sketch_acc": "Sketch accuracy"},
                       os.path.join(fig_dir, f"task3_study_{study}.png"), log_x=log_x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pacs-root", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--protocol", default="shared/splits/pacs_sketch_seed6304.json")
    parser.add_argument("--cache", default="/content/pacs256_with_sketch.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fig-dir", default="report/figures")
    parser.add_argument("--blind", action="store_true", help="hide all result printouts")
    main(parser.parse_args())
