from __future__ import annotations

"""SAM -- Sharpness-Aware Minimization (Task 3, Step 3): min_theta max_{||eps||<=rho} L(theta+eps).

`SAMOptimizer` below follows the standard two-step public SAM pattern (ascent step, then a
descent step at the perturbed point using the original optimizer) -- attribute this pattern
to Foret et al. (2021) / the widely-used community reference implementation in the README.

TODO (Task 3, Day 4): reuse `task2.methods.source_only`'s domain-balanced batch construction,
but replace the single optimizer.step() with:
    freeze_batchnorm_stats(model)             # keep BN running stats frozen for BOTH passes
    loss = criterion(model(images), labels); loss.backward()
    optimizer.first_step(zero_grad=True)
    criterion(model(images), labels).backward()
    optimizer.second_step(zero_grad=True)
rho = 0.05 for the main comparison; controlled study sweeps rho in {0.01, 0.05, 0.1}.
"""

from typing import Type

import torch


class SAMOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls: Type[torch.optim.Optimizer], rho: float = 0.05, **base_optimizer_kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack(
                [
                    p.grad.norm(2).to(device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            2,
        )

    def step(self, closure=None):
        raise RuntimeError("Use first_step()/second_step() explicitly; SAM needs two forward/backward passes.")


RHO_MAIN = 0.05


def train_sam(*args, **kwargs):
    raise NotImplementedError("Implement per the module docstring on Task 3's implementation day.")
