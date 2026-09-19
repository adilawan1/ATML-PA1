from __future__ import annotations

"""Freeze a trained checkpoint and cache its penultimate features + logits for CIFAR-10
train/val/test and both CIFAR-100 unknown groups, so every downstream score (MSP/MLS/Energy/
Mahalanobis, and later evaluation) reads identical, already-computed outputs.

TODO (Task 4, Day 5): for a given checkpoint, run a no-grad forward pass over each split via
`task4.models.resnet_cifar.penultimate_features` + `model.fc`, and save
{"features": ..., "logits": ..., "labels": ...} tensors to `task4/cache/<method>_<split>.pt`
(git-ignored; see task4/cache/.gitkeep).
"""

if __name__ == "__main__":
    raise NotImplementedError("Fill in per the module docstring once a Task 4 checkpoint exists.")
