from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

from sklearn.model_selection import train_test_split

from shared.pacs import list_domain_files

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
SPLIT_SEED = 6304
DEFAULT_PROTOCOL_PATH = "shared/splits/pacs_sketch_seed6304.json"


def build_pacs_protocol(root: str, out_path: str = DEFAULT_PROTOCOL_PATH) -> Dict:
    """Build the single PACS split protocol shared by Task 2 and Task 3.

    Stratified 80/20 train/val split per source domain (seed 6304); the complete target
    (Sketch) domain is recorded separately and must only be read by Task 2's adaptation step
    or by each task's final evaluation script -- never by Task 3 training/selection code.
    """
    protocol: Dict = {
        "seed": SPLIT_SEED,
        "source_domains": SOURCE_DOMAINS,
        "target_domain": TARGET_DOMAIN,
    }

    for domain in SOURCE_DOMAINS:
        samples = list_domain_files(root, domain)
        paths = [p for p, _ in samples]
        labels = [label for _, label in samples]
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=0.2, random_state=SPLIT_SEED, stratify=labels
        )
        protocol[domain] = {
            "train": list(zip(train_paths, train_labels)),
            "val": list(zip(val_paths, val_labels)),
        }

    protocol[TARGET_DOMAIN] = {"all": list_domain_files(root, TARGET_DOMAIN)}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(protocol, f, indent=2)
    return protocol


def load_pacs_protocol(path: str = DEFAULT_PROTOCOL_PATH) -> Dict:
    with open(path) as f:
        return json.load(f)


def validate_protocol(protocol: Dict, root: str) -> int:
    """Number of files listed in the split that are missing under `root` (0 = the split fits this copy of PACS)."""
    listed = [p for d in SOURCE_DOMAINS for k in ("train", "val") for p, _ in protocol[d][k]] + [p for p, _ in protocol[TARGET_DOMAIN]["all"]]
    return sum(not os.path.exists(os.path.join(root, p)) for p in listed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="PACS root: <root>/<domain>/<class>/<image>")
    parser.add_argument("--out", default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--regenerate", action="store_true", help="rebuild the split even if one is already committed")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.regenerate:
        # The committed split is the authoritative one: every session and every task uses exactly these files,
        # independent of the local scikit-learn version. Just check it fits this copy of the dataset.
        missing = validate_protocol(load_pacs_protocol(args.out), args.root)
        if missing:
            raise SystemExit(f"{args.out} lists {missing} files that are not under {args.root}; rebuild PACS with shared.prepare_pacs")
        print(f"Using the committed split {args.out} (all listed files present under {args.root})")
    else:
        build_pacs_protocol(args.root, args.out)
        print(f"Wrote PACS protocol to {args.out}")
