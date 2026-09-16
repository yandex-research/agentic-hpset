"""Lookahead optimizer wrapper around AdamW.

Maintains a slow copy of parameters; every `k` fast steps the slow weights
are interpolated toward the fast weights with mixing coefficient `alpha`,
and the fast weights are reset to match the slow ones. This stabilizes the
optimizer trajectory and often improves generalization. The slow-weight
maintenance happens inside `step()` so the training loop sees a regular
optimizer interface.

Reference: Zhang et al. 2019.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


class Lookahead(torch.optim.Optimizer):
    def __init__(self, base: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        if not isinstance(base, torch.optim.Optimizer):
            raise TypeError("Lookahead requires a torch.optim.Optimizer base.")
        if k < 1:
            raise ValueError("Lookahead k must be >= 1.")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Lookahead alpha must be in (0, 1].")
        self._base = base
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        self._slow: list[torch.Tensor] = []
        for group in self._base.param_groups:
            for p in group["params"]:
                self._slow.append(p.detach().clone())
        self.defaults = self._base.defaults

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._base.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        self._base.param_groups = value

    @property
    def state(self) -> dict[Any, Any]:
        return self._base.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = self._base.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            i = 0
            for group in self._base.param_groups:
                for p in group["params"]:
                    slow = self._slow[i]
                    slow.mul_(1.0 - self.alpha).add_(p.detach(), alpha=self.alpha)
                    p.copy_(slow)
                    i += 1
        return loss

    def state_dict(self) -> dict[str, Any]:
        return self._base.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._base.load_state_dict(state_dict)


def build_lookahead_v0(
    model: torch.nn.Module,
    trial_params: dict[str, Any],
) -> torch.optim.Optimizer:
    base = torch.optim.AdamW(
        model.parameters(),
        lr=float(trial_params["lr"]),
        weight_decay=float(trial_params["weight_decay"]),
    )
    return Lookahead(base, k=5, alpha=0.5)
