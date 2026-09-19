from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18_backbone(num_classes: int = 7) -> nn.Module:
    """ImageNet-pretrained ResNet-18 with its classifier replaced by a `num_classes` head.

    Reused unchanged by Task 3 (same architecture/initialization/preprocessing protocol).
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_batchnorm_stats(model: nn.Module) -> None:
    """Call after model.train(): puts BatchNorm modules in eval mode so running mean/var stay
    at their pretrained ImageNet values, per the assignment's BatchNorm policy for Tasks 2/3.
    gamma/beta remain trainable since only running-stats update is disabled.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def penultimate_features(model: nn.Module, x):
    """512-d feature immediately before `model.fc` -- the MMD/discriminator input."""
    m = model
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)
    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)
    x = m.avgpool(x)
    return x.flatten(1)
