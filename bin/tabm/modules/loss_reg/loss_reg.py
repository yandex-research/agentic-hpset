from __future__ import annotations

import torch
import torch.nn.functional as F

from bin.tabm.core import Dataset


def loss_reg_v0(
    preds: torch.Tensor, targets: torch.Tensor, _dataset_meta: Dataset
) -> torch.Tensor:
    return F.mse_loss(preds, targets)


__all__ = ["loss_reg_v0"]
