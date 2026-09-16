"""Materialized optimizer implementation."""

from __future__ import annotations

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


_implementation = build_adamw


def build_adamw(*args: Any, **kwargs: Any) -> Any:
    return _implementation(*args, **kwargs)
