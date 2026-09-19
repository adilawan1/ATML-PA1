from __future__ import annotations

import argparse

import yaml

from common.seed import set_seed
from task4.methods.gcsc import train_gcsc
from task4.methods.vanilla import train_vanilla

METHOD_TRAINERS = {
    "vanilla": train_vanilla,
    "gcsc": train_gcsc,
    # "proser": train_proser,  # wire up once task4/methods/proser.py's train_proser is implemented
    # "rpl": train_rpl,        # optional
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Task 4 OSR method from a config file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True, help="CIFAR-10/100 download/cache root")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 6304))
    method = config["method"]
    if method not in METHOD_TRAINERS:
        raise NotImplementedError(f"Method '{method}' is not wired into task4/train.py yet.")

    METHOD_TRAINERS[method](
        data_root=args.data_root,
        split_path=config["data"]["split_path"],
        checkpoint_path=config["output"]["checkpoint"],
        metrics_path=config["output"]["metrics"],
    )


if __name__ == "__main__":
    main()
