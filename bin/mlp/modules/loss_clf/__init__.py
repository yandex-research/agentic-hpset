from __future__ import annotations

import torch
import torch.nn.functional as F

from bin.mlp.core import Dataset

from .label_smooth import loss_clf_v1


def loss_clf_v0(preds: torch.Tensor, targets: torch.Tensor, dataset_meta: Dataset) -> torch.Tensor:
    if dataset_meta.is_binclass:
        return F.binary_cross_entropy_with_logits(preds, targets)
    if dataset_meta.is_multiclass:
        return F.cross_entropy(preds, targets)
    raise RuntimeError(f"Classification loss does not support task_type={dataset_meta.task_type!r}.")


LOSS_CLF_MAP = {0: loss_clf_v0, 1: loss_clf_v1}

__all__ = ["loss_clf_v0", "loss_clf_v1", "LOSS_CLF_MAP"]
