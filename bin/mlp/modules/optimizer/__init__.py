from __future__ import annotations

from typing import Any

import torch

from .lookahead import Lookahead, build_lookahead_v0


def build_adamw_v0(
    model: torch.nn.Module,
    trial_params: dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(trial_params["lr"]),
        weight_decay=float(trial_params["weight_decay"]),
    )


OPTIMIZER_MAP = {0: build_adamw_v0, 1: build_lookahead_v0}

__all__ = ["build_adamw_v0", "build_lookahead_v0", "Lookahead", "OPTIMIZER_MAP"]
