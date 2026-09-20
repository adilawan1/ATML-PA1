from __future__ import annotations

import argparse
import sys
from collections import Counter

from shared.pacs import CLASSES, DOMAINS, list_domain_files

# Image counts of the standard PACS release (Li et al., 2017).
EXPECTED_DOMAIN_COUNTS = {"photo": 1670, "art_painting": 2048, "cartoon": 2344, "sketch": 3929}


def verify(root: str) -> bool:
    ok = True
    for domain in DOMAINS:
        try:
            samples = list_domain_files(root, domain)
        except FileNotFoundError as err:
            print(f"[MISSING] {err}")
            ok = False
            continue
        per_class = Counter(label for _, label in samples)
        expected = EXPECTED_DOMAIN_COUNTS[domain]
        status = "ok" if len(samples) == expected else f"MISMATCH (expected {expected})"
        print(f"{domain:13s} {len(samples):5d} images  [{status}]")
        print("              " + ", ".join(f"{CLASSES[i]}={per_class[i]}" for i in range(len(CLASSES))))
        ok &= len(samples) == expected
    print("\nPACS layout OK" if ok else "\nPACS layout has problems -- see above")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a PACS copy has the folder layout/counts the loaders expect.")
    parser.add_argument("--root", required=True, help="Directory whose subfolders are photo/art_painting/cartoon/sketch")
    sys.exit(0 if verify(parser.parse_args().root) else 1)
