# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch


class TargetEncodedCategoricalEmbedding(nn.Module):
    """One-hot concatenated with smoothed target-mean encoding per categorical feature.

    For each feature, output is [one_hot(x_i)] || [target_table[x_i]] flattened.
    The target table is precomputed from train and registered as a buffer.
    """

    def __init__(
        self, cardinalities: list[int], target_tables: list[np.ndarray]
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        n_classes = target_tables[0].shape[1]
        self.n_classes = n_classes
        for i, tab in enumerate(target_tables):
            self.register_buffer(f'te_{i}', torch.as_tensor(tab, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, cardinality in enumerate(self.cardinalities):
            idx = x[:, i].clamp(0, cardinality - 1)
            pieces.append(F.one_hot(idx, cardinality).float())
            table: Tensor = getattr(self, f'te_{i}')
            pieces.append(table[idx])
        return torch.cat(pieces, dim=1)


def _smoothed_target_table(
    cat_train: np.ndarray,
    y_train: np.ndarray,
    cardinality: int,
    n_classes: int,
    smoothing: float | None = None,
) -> np.ndarray:
    """Return a conservatively smoothed target-mean encoding table.

    The row reserved by the package for unknown categories stays at the global
    prior. Subtracting the current row before weighting removes singleton
    self-leakage and strongly shrinks low-count categories.
    """
    global_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    global_prior = (global_counts + 1e-06) / (global_counts.sum() + 1e-06 * n_classes)
    table = np.tile(global_prior, (cardinality, 1))
    prior_strength = float(
        max(20.0, np.sqrt(max(y_train.shape[0], 1))) if smoothing is None else smoothing
    )
    for c in range(cardinality - 1):
        mask = cat_train == c
        n = int(mask.sum())
        if n == 0:
            continue
        local_counts = np.bincount(y_train[mask], minlength=n_classes).astype(
            np.float64
        )
        local_prob = local_counts / max(n, 1)
        effective_count = max(n - 1, 0)
        weight = effective_count / (effective_count + prior_strength)
        table[c] = weight * local_prob + (1.0 - weight) * global_prior
    table[-1] = global_prior
    return table


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    if dataset_meta.x_cat is None:
        return (None, 0, {'cardinalities': []})
    cat_train = dataset_meta.x_cat['train']
    y_train = np.asarray(dataset_meta.y['train']).reshape(-1).astype(np.int64)
    n_classes = int(dataset_meta.n_classes)
    target_tables = [
        _smoothed_target_table(cat_train[:, i], y_train, cardinality, n_classes)
        for i, cardinality in enumerate(cat_cardinalities)
    ]
    output_dim = sum(cat_cardinalities) + n_classes * len(cat_cardinalities)
    return (
        TargetEncodedCategoricalEmbedding(cat_cardinalities, target_tables),
        output_dim,
        {'cardinalities': cat_cardinalities, 'n_classes': n_classes},
    )
