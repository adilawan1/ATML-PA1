from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 32x32 CIFAR images: 3x3/stride-1 stem, no initial max-pool
    (replaces the ImageNet 7x7/stride-2 conv + max-pool), per the assignment's Task 4 spec.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def penultimate_features(model: nn.Module, x):
    """512-d feature immediately before `model.fc`."""
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


def split_forward_before_layer3(model: nn.Module, x):
    """Forward pass up to (and including) layer2 -- the manifold-mixup point PROSER needs."""
    m = model
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)
    x = m.layer1(x)
    return m.layer2(x)


def forward_from_layer3(model: nn.Module, h):
    """Complete the forward pass from a layer2 output `h`, returning (feature, logits)."""
    m = model
    x = m.layer3(h)
    x = m.layer4(x)
    x = m.avgpool(x)
    feat = x.flatten(1)
    return feat, m.fc(feat)
