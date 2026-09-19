from __future__ import annotations

import random

import numpy as np
import torch

DEFAULT_SEED = 6304


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seeded_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
