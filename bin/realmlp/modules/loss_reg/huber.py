"""RealMLP regression loss (v1): Huber loss on standardized targets.

Hypothesis: the per-member target preprocessing standardizes by mean/std, so
a heavy-tailed target inflates the std, compresses the bulk of targets near
zero, and leaves the few outliers dominating the MSE gradient — the net
underfits the bulk. Huber (Huber 1964) caps the outlier gradient; robust
objectives are shipped-standard in GBDT libraries (LightGBM/XGBoost) for
exactly this regime. This engages the robust-loss caution directly: on clean
benchmarks a robust loss loses to MSE (RMSE is the metric), so the module is
only expected to win on heavy-tailed-target datasets, where the per-dataset
tuner selects it.

Mechanism: per-element Huber with ``huber_delta`` (default 1.0 — quadratic
for the standardized bulk, linear beyond one std-unit of residual), then the
identical v0 reduction: mean over output dims and batch per member, summed
across members.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def regression_loss_step_huber(
    logits: torch.Tensor,
    target: torch.Tensor,
    cfg: Dict[str, Any],
    _t: float,
) -> torch.Tensor:
    """``logits`` shape ``(n_ens, batch, out_dim)``; ``target`` same shape."""
    delta = float(cfg.get("huber_delta", 1.0))
    elementwise = F.huber_loss(logits, target, delta=delta, reduction="none")
    member_losses = elementwise.mean(dim=-1)
    return member_losses.mean(dim=1).sum()


__all__ = ["regression_loss_step_huber"]
