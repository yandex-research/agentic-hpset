from __future__ import annotations

import torch
import torch.nn.functional as F

from bin.tabm.core import Dataset


def loss_clf_v0(
    preds: torch.Tensor, targets: torch.Tensor, dataset_meta: Dataset
) -> torch.Tensor:
    if dataset_meta.is_binclass:
        return F.binary_cross_entropy_with_logits(preds, targets.float())
    if dataset_meta.is_multiclass:
        if preds.ndim == 3:
            return F.cross_entropy(preds.flatten(0, 1), targets.reshape(-1))
        return F.cross_entropy(preds, targets)
    raise RuntimeError(
        f"Classification loss does not support task_type={dataset_meta.task_type!r}."
    )


def loss_clf_v1(
    preds: torch.Tensor, targets: torch.Tensor, dataset_meta: Dataset
) -> torch.Tensor:
    label_smoothing = 0.1
    if dataset_meta.is_binclass:
        smooth_targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
        return F.binary_cross_entropy_with_logits(preds, smooth_targets)
    if dataset_meta.is_multiclass:
        if preds.ndim == 3:
            return F.cross_entropy(
                preds.flatten(0, 1),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
            )
        return F.cross_entropy(preds, targets, label_smoothing=label_smoothing)
    raise RuntimeError(
        f"Classification loss does not support task_type={dataset_meta.task_type!r}."
    )


LOSS_CLF_MAP = {0: loss_clf_v0, 1: loss_clf_v1}

__all__ = ["loss_clf_v0", "loss_clf_v1", "LOSS_CLF_MAP"]
