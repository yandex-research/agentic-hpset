"""RealMLP classification loss (v0).

Per-sample cross-entropy with optionally scheduled label smoothing
(``use_ls`` + ``ls_eps`` scaled by ``ls_eps_sched(t)``), reshaped back to
per-ensemble-member losses, averaged within each member, then summed across
members so that each member receives an equal-weight gradient signal.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .._schedules import schedule_value

def classification_loss_step_v0(
    logits: torch.Tensor,
    y_idx: torch.Tensor,
    cfg: Dict[str, Any],
    t: float,
) -> torch.Tensor:
    """``logits`` shape ``(n_ens, batch, n_classes)``; ``y_idx`` shape ``(n_ens, batch)``."""
    eps = float(cfg.get("ls_eps", 0.0)) * schedule_value(cfg.get("ls_eps_sched", "constant"), t)
    eps = eps if cfg.get("use_ls", False) else 0.0
    y_flat = y_idx.reshape(-1)
    member_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y_flat,
        label_smoothing=eps,
        reduction="none",
    ).reshape(logits.shape[0], -1)
    return member_losses.mean(dim=1).sum()

__all__ = ["classification_loss_step_v0"]
