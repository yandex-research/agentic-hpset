from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


class Lion(torch.optim.Optimizer):
    """Lion optimizer: sign-of-momentum updates with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        super().__init__(
            params, {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                if weight_decay != 0.0:
                    update.add_(p, alpha=weight_decay)
                p.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


def build_lion_v1(
    model: torch.nn.Module,
    trial_params: dict[str, Any],
) -> torch.optim.Optimizer:
    return Lion(
        model.parameters(),
        lr=float(trial_params["lr"]) / 3.0,
        weight_decay=float(trial_params["weight_decay"]) * 3.0,
    )


__all__ = ["Lion", "build_lion_v1"]
