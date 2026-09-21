from __future__ import annotations

"""SAM -- Sharpness-Aware Minimization (Task 3, Step 3): min_theta max_{||eps||_2<=rho} L_ERM(theta + eps).

Standard, non-adaptive SAM with AdamW as the base optimizer (same lr / weight decay as ERM). Each step
does two forward/backward passes: (1) the gradient at theta gives the normalized ascent perturbation
eps = rho * g / ||g||; (2) the gradient at theta + eps is used to update the ORIGINAL parameters.
BatchNorm running statistics stay frozen during both passes (the trainer keeps BN modules in eval mode).

`SAMOptimizer` follows the standard public two-step SAM pattern (Foret et al., 2021; cf. the widely used
davda54/sam implementation) -- attributed in the README.
"""

from typing import Dict, Type

import torch
import torch.nn.functional as F

from shared.trainer import Batch, Method


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
            torch.stack([p.grad.norm(2).to(device) for group in self.param_groups for p in group["params"] if p.grad is not None]),
            2,
        )

    def step(self, closure=None):
        raise RuntimeError("Use first_step()/second_step() explicitly; SAM needs two forward/backward passes.")


class SAM(Method):
    uses_target = False

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)
        self.rho = cfg["sam"]["rho"]

    def build_optimizer(self, params, lr: float, weight_decay: float):
        return SAMOptimizer(params, torch.optim.AdamW, rho=self.rho, lr=lr, weight_decay=weight_decay)

    def step(self, batch: Batch, optimizer: SAMOptimizer, progress: float) -> Dict[str, torch.Tensor]:
        loss = F.cross_entropy(self.model(batch.source_x), batch.source_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        perturbed_loss = F.cross_entropy(self.model(batch.source_x), batch.source_y)
        perturbed_loss.backward()
        optimizer.second_step(zero_grad=True)
        return {"cls_loss": loss.detach(), "perturbed_loss": perturbed_loss.detach()}
