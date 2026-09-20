from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def train_linear_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    seed: int = 6304,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 128,
) -> Tuple[nn.Linear, Dict[str, float]]:
    """Linear classifier on frozen features: AdamW, at most `max_epochs`, early-stopped after
    `patience` epochs without a strictly better validation accuracy (best weights restored).
    No feature standardization -- the assignment specifies the head is trained on the raw
    frozen representation.
    """
    train_y, val_y = train_y.long(), val_y.long()  # STL-10 labels arrive as uint8
    torch.manual_seed(seed)
    head = nn.Linear(train_x.shape[1], num_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    best_acc, best_epoch, best_state, bad_epochs = -1.0, -1, None, 0
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        head.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            loss = criterion(head(train_x[idx]), train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_acc = (head(val_x).argmax(dim=1) == val_y).float().mean().item()

        if val_acc > best_acc:
            best_acc, best_epoch, bad_epochs = val_acc, epoch, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_val_acc": best_acc, "best_epoch": best_epoch, "epochs_run": epochs_run}
