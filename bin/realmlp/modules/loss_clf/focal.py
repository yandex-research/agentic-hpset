"""RealMLP classification loss (v1): focal loss.

Hypothesis: v0's meta-tuned CE (+ scheduled label smoothing) optimizes the
benchmark average; on class-imbalanced datasets the loss is dominated by the
easy majority class. Focal loss (Lin et al. 2017) reweights per-sample CE by
``(1 - p_t)^gamma``, concentrating gradient signal on hard/minority samples —
standard for imbalance far beyond its detection origins. The per-dataset
tuner routes imbalanced tasks here; on balanced data it underweights easy
examples and slows fitting, so it is deselected there.

Mechanism: per-sample focal term with ``focal_gamma`` (default 1.5), then the
identical v0 reduction — reshape to per-member losses, mean within members,
sum across members. Replaces label smoothing rather than stacking on it.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def classification_loss_step_focal(
    logits: torch.Tensor,
    y_idx: torch.Tensor,
    cfg: Dict[str, Any],
    t: float,
) -> torch.Tensor:
    """``logits`` shape ``(n_ens, batch, n_classes)``; ``y_idx`` shape ``(n_ens, batch)``."""
    gamma = float(cfg.get("focal_gamma", 1.5))
    y_flat = y_idx.reshape(-1)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_flat, reduction="none")
    p_t = torch.exp(-ce)
    member_losses = ((1.0 - p_t) ** gamma * ce).reshape(logits.shape[0], -1)
    return member_losses.mean(dim=1).sum()


__all__ = ["classification_loss_step_focal"]
