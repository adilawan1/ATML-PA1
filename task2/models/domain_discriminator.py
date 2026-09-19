from __future__ import annotations

import math

import torch.nn as nn
from torch.autograd import Function


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grl(x, alpha: float):
    """Gradient-reversal layer: identity on the forward pass, scaled-negated gradient on backward."""
    return _GradientReversal.apply(x, alpha)


def grl_schedule(progress: float, gamma: float = 10.0) -> float:
    """Ganin et al. 2016 schedule: alpha(p) = 2 / (1 + exp(-gamma * p)) - 1, p in [0, 1]."""
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


class DomainDiscriminator(nn.Module):
    """Binary source/target discriminator shared by DANN and CDAN.

    DANN feeds it the 512-d feature `f`; CDAN feeds it `g(x) = vec(f (x) p)` (see
    `task2/methods/cdan.py`), so `input_dim` differs between the two but the architecture
    (256-unit hidden layer, ReLU, dropout 0.5, 2-class output) does not.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)
