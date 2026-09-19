from __future__ import annotations

"""PROSER -- classifier and data placeholders (Task 4, Step 4).

Initializes from the selected Vanilla checkpoint, appends 5 randomly initialized dummy
classifiers, and fine-tunes for 50 epochs (SGD, lr=1e-3, momentum 0.9, wd=5e-4, cosine,
batch 128, seed 6304). Two losses, each half of every mini-batch:
  - classifier placeholders (beta=1): on ordinary examples, correct class stays largest, but
    with it excluded one dummy classifier should become the strongest remaining response
    (loss defined in Zhou et al. 2021).
  - data placeholders (gamma=0.1): `task4.methods.manifold_mixup.manifold_mixup` on the
    layer2/layer3 boundary (`task4.models.resnet_cifar.split_forward_before_layer3` /
    `forward_from_layer3`) between two different-class examples, trained toward the dummy
    classifiers rather than either original class.

No CIFAR-100 image participates in either placeholder. Checkpoint selection uses CIFAR-10
validation accuracy only. Evaluate with (a) MLS over the 10 known-class logits, for direct
comparison with Vanilla/GCSC, and (b) the placeholder-based detection score combining the
strongest dummy response with the known-class responses, per the reference implementation.

TODO (Task 4, Day 5): read Zhou et al. (2021) closely for the exact classifier-placeholder
loss and the placeholder detection score before implementing -- both are specified precisely
enough in the paper that an approximation here would misrepresent the method.
"""

import torch.nn as nn


class DummyClassifierHead(nn.Module):
    """Appends `num_dummy` extra output units alongside the original `num_classes` logits."""

    def __init__(self, feature_dim: int, num_classes: int, num_dummy: int = 5):
        super().__init__()
        self.known_fc = nn.Linear(feature_dim, num_classes)
        self.dummy_fc = nn.Linear(feature_dim, num_dummy)

    def forward(self, features):
        return self.known_fc(features), self.dummy_fc(features)


BETA_CLASSIFIER_PLACEHOLDER = 1.0
GAMMA_DATA_PLACEHOLDER = 0.1
NUM_DUMMY_CLASSIFIERS = 5


def train_proser(*args, **kwargs):
    raise NotImplementedError("Implement per the module docstring on Task 4's implementation day.")
