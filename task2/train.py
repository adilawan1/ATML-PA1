from __future__ import annotations

import argparse

import yaml

from common.seed import set_seed
from shared.pacs_protocol import load_pacs_protocol
from task2.methods.source_only import train_source_only

METHOD_TRAINERS = {
    "source_only": train_source_only,
    # "dan": train_dan,      # wire up once task2/methods/dan.py is implemented
    # "dann": train_dann,    # wire up once task2/methods/dann.py is implemented
    # "cdan": train_cdan,    # wire up once task2/methods/cdan.py is implemented
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Task 2 UDA method from a config file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pacs-root", default=None, help="Overrides data.pacs_root from the config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 6304))
    protocol = load_pacs_protocol(config["data"]["protocol_path"])

    method = config["method"]
    if method not in METHOD_TRAINERS:
        raise NotImplementedError(f"Method '{method}' is not wired into task2/train.py yet.")

    pacs_root = args.pacs_root or config["data"]["pacs_root"]
    if pacs_root is None:
        raise ValueError("Pass --pacs-root (or set data.pacs_root in the config).")

    train_source_only(
        protocol,
        pacs_root,
        image_size=config["data"]["image_size"],
        crop_size=config["data"]["crop_size"],
        batch_per_domain=config["train"]["batch_per_source_domain"],
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        max_epochs=config["train"]["max_epochs"],
        patience=config["train"]["patience"],
        checkpoint_path=config["output"]["checkpoint"],
        metrics_path=config["output"]["metrics"],
    )


if __name__ == "__main__":
    main()
