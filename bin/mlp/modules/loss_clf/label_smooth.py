"""Label-smoothed classification loss.

Replaces the standard one-hot target with a smoothed target where the true
class gets `1 - eps` mass and the remainder is uniformly distributed over
the other classes (`eps = 0.1`). Label smoothing prevents the network from
producing arbitrarily large logits, regularizes overconfident predictions,
and is a known generalization boost on tabular classification.

- Binclass: smoothed BCE-with-logits target via `targets * (1-eps) + 0.5*eps`.
- Multiclass: `F.cross_entropy(..., label_smoothing=eps)`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bin.mlp.core import Dataset


_EPS = 0.1


def loss_clf_v1(preds: torch.Tensor, targets: torch.Tensor, dataset_meta: Dataset) -> torch.Tensor:
    if dataset_meta.is_binclass:
        soft = targets * (1.0 - _EPS) + 0.5 * _EPS
        return F.binary_cross_entropy_with_logits(preds, soft)
    if dataset_meta.is_multiclass:
        return F.cross_entropy(preds, targets, label_smoothing=_EPS)
    raise RuntimeError(
        f"Label-smoothing classification loss does not support task_type={dataset_meta.task_type!r}."
    )
