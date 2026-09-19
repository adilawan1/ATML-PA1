from __future__ import annotations

"""DANN -- adversarial domain alignment via gradient reversal (Task 2, Step 3).

Discriminator: 256-unit hidden layer, ReLU, dropout 0.5, 2-class output, attached to the
512-d feature (see `task2.models.domain_discriminator.DomainDiscriminator`). Schedule
alpha(p) = 2/(1+exp(-10p)) - 1 via `grl_schedule`/`grl`. Only source examples contribute to
the classification loss; both source and (unlabeled) target examples contribute to the
domain loss, with unit weight.

TODO (Task 2, Day 3): per training step, compute p = global_step / total_steps, alpha =
grl_schedule(p, gamma=10.0), run `grl(penultimate_features(model, x), alpha)` through the
discriminator for pooled source+target features, and add the domain cross-entropy
(source label 0, target label 1) to the source classification loss.
"""

from task2.models.backbone import penultimate_features  # noqa: F401
from task2.models.domain_discriminator import DomainDiscriminator, grl, grl_schedule  # noqa: F401

FEATURE_DIM = 512
DOMAIN_LOSS_WEIGHT = 1.0
GRL_GAMMA = 10.0


def train_dann(*args, **kwargs):
    raise NotImplementedError("Implement per the module docstring on Task 2's implementation day.")
