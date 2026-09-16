"""RealMLP regression loss (v0): per-member MSE averaged then summed.

``logits`` shape ``(n_ens, batch, out_dim)``; ``target`` same shape (already
broadcast across the ensemble dim by the train loop).
"""

from __future__ import annotations

from typing import Any, Dict

import torch

def regression_loss_step_v0(
    logits: torch.Tensor,
    target: torch.Tensor,
    _cfg: Dict[str, Any],
    _t: float,
) -> torch.Tensor:
    member_losses = (logits - target).square().mean(dim=-1)
    return member_losses.mean(dim=1).sum()

__all__ = ["regression_loss_step_v0"]
