from __future__ import annotations

import argparse

import yaml

from common.seed import set_seed
from shared.pacs_protocol import load_pacs_protocol
from task3.methods.erm import load_erm_checkpoint

METHOD_TRAINERS = {
    # "dan_dg": train_dan_dg,  # wire up once task3/methods/dan_dg.py is implemented
    # "sam": train_sam,       # wire up once task3/methods/sam.py's train_sam is implemented
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Task 3 DG method from a config file.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 6304))

    if config["method"] == "erm":
        # No Sketch access, no retraining: just verify the shared Task 2 checkpoint loads.
        load_erm_checkpoint(config["reuse_checkpoint"])
        print(f"ERM baseline = Task 2 checkpoint at {config['reuse_checkpoint']} (not retrained).")
        return

    # IMPORTANT: only load the source-side protocol here. `load_pacs_protocol` returns the
    # target split too, but this script must never read protocol["sketch"] before
    # evaluate_sketch.py runs -- keep that access confined to the evaluation script.
    protocol = load_pacs_protocol(config["data"]["protocol_path"])

    method = config["method"]
    if method not in METHOD_TRAINERS:
        raise NotImplementedError(f"Method '{method}' is not wired into task3/train.py yet.")

    METHOD_TRAINERS[method](protocol, config)


if __name__ == "__main__":
    main()
