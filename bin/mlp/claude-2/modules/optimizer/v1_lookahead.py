"""Materialized optimizer implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import torch


def build_adamw(
    model: torch.nn.Module, trial_params: dict[str, Any]
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(trial_params['lr']),
        weight_decay=float(trial_params['weight_decay']),
    )


class Lion(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            params, {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                momentum = state.setdefault('momentum', torch.zeros_like(parameter))
                if group['weight_decay']:
                    parameter.mul_(1.0 - group['lr'] * group['weight_decay'])
                update = momentum.mul(beta1).add(gradient, alpha=1.0 - beta1)
                parameter.add_(update.sign(), alpha=-group['lr'])
                momentum.mul_(beta2).add_(gradient, alpha=1.0 - beta2)
        return loss


class Lookahead(torch.optim.Optimizer):
    def __init__(self, base: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.base = base
        self.k = k
        self.alpha = alpha
        self._step_count_lookahead = 0
        defaults = dict(base.defaults)
        super().__init__(base.param_groups, defaults)
        self.param_groups = base.param_groups
        self.state = base.state
        self.slow = [p.detach().clone() for g in self.param_groups for p in g['params']]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base.step(closure)
        self._step_count_lookahead += 1
        if self._step_count_lookahead % self.k == 0:
            for slow, parameter in zip(
                self.slow,
                (p for group in self.param_groups for p in group['params']),
                strict=True,
            ):
                slow.add_(parameter - slow, alpha=self.alpha)
                parameter.copy_(slow)
        return loss


def make_optimizer_variant(
    strategy: str, params: dict[str, Any], symbol: str
) -> Callable[[torch.nn.Module, dict[str, Any]], torch.optim.Optimizer]:
    def build(
        model: torch.nn.Module, trial_params: dict[str, Any]
    ) -> torch.optim.Optimizer:
        lr = float(trial_params['lr'])
        weight_decay = float(trial_params['weight_decay'])
        if strategy == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=float(params.get('momentum', 0.9)),
                nesterov=bool(params.get('nesterov', True)),
                weight_decay=weight_decay,
            )
        if strategy == 'lion':
            return Lion(
                model.parameters(),
                lr=lr * float(params.get('lr_scale', 1.0)),
                weight_decay=weight_decay
                * float(params.get('weight_decay_scale', 1.0)),
            )
        if strategy == 'lookahead':
            return Lookahead(
                build_adamw(model, trial_params),
                k=int(params.get('k', 5)),
                alpha=float(params.get('alpha', 0.5)),
            )
        return build_adamw(model, trial_params)

    build.__name__ = f'optimizer_{strategy}_{abs(hash(symbol)) & 0xFFFF:x}'
    build.__doc__ = f'Ported from {symbol}; extracted optimizer: {strategy}.'
    return build


_implementation = make_optimizer_variant(
    'lookahead',
    {'alpha': 0.5, 'k': 5},
    'train.train_lookahead.train_v6',
)


def optimizer_lookahead_v1(*args: Any, **kwargs: Any) -> Any:
    return _implementation(*args, **kwargs)
