from __future__ import annotations

import argparse

from task4.data.cifar10 import build_cifar10_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    build_cifar10_split(args.data_root)
    print(
        "CIFAR-10 90/10 split written. CIFAR-100 unknown groups are fixed and need no split "
        "file -- see task4/data/cifar100_unknowns.py (NEAR_UNKNOWN / FAR_UNKNOWN)."
    )
