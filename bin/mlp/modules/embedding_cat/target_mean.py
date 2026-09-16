"""One-hot categorical embedding augmented with smoothed target-mean encoding.

For each categorical column, a per-category smoothed target mean is computed
on the training labels (Bayesian smoothing toward the global mean). For
multiclass tasks the smoothed per-class probability vector is used (with the
last class dropped to avoid linear redundancy). These target statistics are
concatenated to the per-column one-hot, injecting a supervised signal
directly into the categorical representation. Built once at construction;
unknown categories (`train_max + 1`) fall back to the prior row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _smoothed_means(
    column: np.ndarray, y: np.ndarray, n_classes: int, smoothing: float = 10.0
) -> np.ndarray:
    cardinality = int(column.max()) + 1 if column.size else 0
    if n_classes == 2:
        prior = float(y.mean()) if y.size else 0.5
        out = np.full((cardinality + 1, 1), prior, dtype=np.float32)
        for c in range(cardinality):
            mask = column == c
            n = int(mask.sum())
            if n:
                out[c, 0] = (y[mask].sum() + smoothing * prior) / (n + smoothing)
        return out
    prior = (
        np.bincount(y, minlength=n_classes).astype(np.float32) if y.size else np.zeros(n_classes, dtype=np.float32)
    )
    prior /= max(prior.sum(), 1.0)
    out = np.tile(prior, (cardinality + 1, 1)).astype(np.float32)
    for c in range(cardinality):
        mask = column == c
        n = int(mask.sum())
        if n:
            counts = np.bincount(y[mask], minlength=n_classes).astype(np.float32)
            out[c] = (counts + smoothing * prior) / (n + smoothing)
    return out[:, :-1].astype(np.float32)


class OneHotPlusTargetMean(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        target_tables: list[Tensor],
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.target_tables = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in target_tables]
        )

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, cardinality in enumerate(self.cardinalities):
            safe_idx = x[:, i].clamp(0, self.target_tables[i].shape[0] - 1)
            pieces.append(F.one_hot(safe_idx, cardinality + 1)[:, :-1].float())
            pieces.append(self.target_tables[i][safe_idx])
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v5(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities or dataset_meta.x_cat is None:
        return None, 0, {"cardinalities": []}
    x_cat_train = dataset_meta.x_cat["train"]
    y_train = dataset_meta.y_raw["train"]
    n_classes = int(dataset_meta.n_classes)
    target_tables: list[Tensor] = []
    extra_dims = 0
    for i, cardinality in enumerate(cat_cardinalities):
        smoothed = _smoothed_means(
            x_cat_train[:, i].astype(np.int64),
            y_train.astype(np.int64) if dataset_meta.is_multiclass else y_train,
            n_classes if dataset_meta.is_multiclass else 2,
        )
        if smoothed.shape[1] == 0:
            smoothed = np.zeros((cardinality + 1, 1), dtype=np.float32)
        target_tables.append(torch.as_tensor(smoothed, device=device, dtype=torch.float32))
        extra_dims += smoothed.shape[1]
    output_dim = sum(cat_cardinalities) + extra_dims
    embedding = OneHotPlusTargetMean(cat_cardinalities, target_tables).to(device)
    return embedding, output_dim, {"cardinalities": cat_cardinalities, "extra_dims": extra_dims}
