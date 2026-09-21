from __future__ import annotations

"""The single training loop behind every Task 2 / Task 3 method (Source-only, DAN, DANN, CDAN,
DAN-DG, SAM), so initialization, source sampling, augmentation, optimizer, BatchNorm policy,
early-stopping rule and budget are identical by construction; only `Method.step` differs.

Per the assignment:
  * batches: 8 examples from each source domain (+24 unlabeled target examples for UDA methods);
  * AdamW lr 1e-4, weight decay 1e-4, at most 30 epochs, stop after 5 epochs without improvement in
    mean source-validation macro-F1; the best-F1 checkpoint is kept (target labels never used);
  * BatchNorm running statistics stay at their pretrained ImageNet values (BN modules are put back
    in eval mode after every `model.train()`); gamma/beta remain trainable;
  * source sampling/augmentation come from a generator that target sampling never touches, so the
    source batch sequence is identical across methods.
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from shared.eval_utils import source_val_metrics
from shared.pacs_data import SOURCE_DOMAINS, TARGET_DOMAIN, BatchStream, PACSData
from task2.models.backbone import build_resnet18_backbone, freeze_batchnorm_stats


@dataclass
class Batch:
    source_x: torch.Tensor  # (n_domains * b, 3, 224, 224), concatenated domain by domain (equal chunks)
    source_y: torch.Tensor
    n_domains: int
    target_x: Optional[torch.Tensor] = None  # unlabeled -- there is deliberately no target_y


class Method:
    uses_target = False

    def __init__(self, cfg: Dict, model: nn.Module, device: str):
        self.cfg, self.model, self.device = cfg, model, device

    def extra_parameters(self):
        return []

    def set_train(self) -> None:
        pass

    def build_optimizer(self, params, lr: float, weight_decay: float):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def step(self, batch: Batch, optimizer, progress: float) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


def load_config(path: str, overrides: Optional[Dict] = None) -> Dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)  # method files hold only their differences from base.yaml
    if base:
        cfg = load_config(os.path.join(os.path.dirname(path), base), cfg)

    def merge(base: Dict, extra: Dict) -> None:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    merge(cfg, overrides or {})
    return cfg


def train_run(
    method_cls: Callable[..., Method],
    cfg: Dict,
    data: PACSData,
    device: str,
    checkpoint_path: str,
    metrics_path: str,
    log: Callable[[str], None] = print,
) -> Dict:
    """Train one method from `cfg`; returns a summary dict and leaves the best checkpoint on disk."""
    tcfg = cfg["train"]
    seed = cfg["seed"]
    torch.manual_seed(seed)  # identical fc initialization for every method
    model = build_resnet18_backbone(num_classes=7).to(device)
    method = method_cls(cfg, model, device)
    if method.uses_target and not data.include_target:
        raise ValueError(f"{method_cls.__name__} needs target images but `data` was built without them")

    optimizer = method.build_optimizer(list(model.parameters()) + list(method.extra_parameters()), tcfg["lr"], tcfg["weight_decay"])

    g_source = torch.Generator().manual_seed(seed)
    g_target = torch.Generator().manual_seed(seed + 1)
    per_domain = tcfg["batch_per_source_domain"]
    streams = {d: BatchStream(len(data.rows(d, "train")), per_domain, g_source) for d in SOURCE_DOMAINS}
    target_stream = BatchStream(len(data.rows(TARGET_DOMAIN, "all")), tcfg["batch_target"], g_target) if method.uses_target else None

    steps_per_epoch = max(len(data.rows(d, "train")) for d in SOURCE_DOMAINS) // per_domain
    total_steps = tcfg["max_epochs"] * steps_per_epoch

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    best_f1, best_epoch, bad_epochs, epochs_run, global_step = -1.0, -1, 0, 0, 0
    started = time.time()
    for epoch in range(tcfg["max_epochs"]):
        epochs_run = epoch + 1
        model.train()
        freeze_batchnorm_stats(model)
        method.set_train()
        sums: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((), device=device))

        for _ in range(steps_per_epoch):
            positions = [streams[d].next() for d in SOURCE_DOMAINS]
            rows = torch.cat([data.rows(d, "train")[pos] for d, pos in zip(SOURCE_DOMAINS, positions)])
            labels = torch.cat([data.labels(d, "train")[pos] for d, pos in zip(SOURCE_DOMAINS, positions)]).to(device)
            source_x = data.batch_train(rows, g_source, device)
            target_x = None
            if target_stream is not None:
                target_rows = data.rows(TARGET_DOMAIN, "all")[target_stream.next()]
                target_x = data.batch_train(target_rows, g_target, device)

            logs = method.step(Batch(source_x, labels, len(SOURCE_DOMAINS), target_x), optimizer, global_step / total_steps)
            for key, value in logs.items():
                sums[key] += value.detach() if torch.is_tensor(value) else value
            global_step += 1

        val = source_val_metrics(model, data, device)
        row = {
            "epoch": epoch,
            **{key: float(value) / steps_per_epoch for key, value in sums.items()},
            "val": val["per_domain"],
            "mean_source_val_accuracy": val["mean_accuracy"],
            "mean_source_val_macro_f1": val["mean_macro_f1"],
            "seconds": round(time.time() - started, 1),
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(
            f"epoch {epoch:2d} | "
            + " ".join(f"{k}={row[k]:.4f}" for k in sums)
            + f" | mean src val F1={val['mean_macro_f1']:.4f} ({row['seconds']:.0f}s)"
        )

        if val["mean_macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val["mean_macro_f1"], epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "mean_source_val_macro_f1": best_f1}, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["patience"]:
                log(f"early stopping after epoch {epoch} (best epoch {best_epoch}, F1 {best_f1:.4f})")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    return {"best_epoch": best_epoch, "best_mean_source_val_macro_f1": best_f1, "epochs_run": epochs_run, "seconds": round(time.time() - started, 1)}
