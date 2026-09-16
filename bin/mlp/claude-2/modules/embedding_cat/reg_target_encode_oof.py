# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np

_N_FOLDS = 5


class _OOFTargetEncoding(nn.Module):
    def __init__(
        self, lookup_tables: list[Tensor], target_mean: float, target_std: float
    ) -> None:
        super().__init__()
        self.n_features = len(lookup_tables)
        self.target_mean = target_mean
        self.target_std = target_std
        for i, table in enumerate(lookup_tables):
            self.register_buffer(f'table_{i}', table)

    def forward(self, x: Tensor) -> Tensor:
        outputs: list[Tensor] = []
        for i in range(self.n_features):
            table = getattr(self, f'table_{i}')
            ids = x[:, i].clamp(min=0, max=table.shape[0] - 1)
            outputs.append(table[ids].unsqueeze(1))
        return torch.cat(outputs, dim=1)


_SMOOTHING = 10.0


def _compute_oof_lookup(
    column: np.ndarray,
    y: np.ndarray,
    size: int,
    n_folds: int,
    smoothing: float,
    prior: float,
    seed: int,
) -> np.ndarray:
    n = column.shape[0]
    if n == 0:
        return np.full(size, prior, dtype=np.float32)
    rng = np.random.RandomState(seed)
    fold_assignment = rng.randint(0, n_folds, size=n)
    sums = np.zeros(size, dtype=np.float64)
    counts = np.zeros(size, dtype=np.float64)
    np.add.at(sums, column, y)
    np.add.at(counts, column, 1.0)
    fold_sums = np.zeros((n_folds, size), dtype=np.float64)
    fold_counts = np.zeros((n_folds, size), dtype=np.float64)
    for fold in range(n_folds):
        mask = fold_assignment == fold
        np.add.at(fold_sums[fold], column[mask], y[mask])
        np.add.at(fold_counts[fold], column[mask], 1.0)
    encoded = np.zeros(n, dtype=np.float64)
    for fold in range(n_folds):
        mask = fold_assignment == fold
        if not mask.any():
            continue
        out_sums = sums - fold_sums[fold]
        out_counts = counts - fold_counts[fold]
        smoothed = (out_sums + smoothing * prior) / (out_counts + smoothing)
        encoded[mask] = smoothed[column[mask]]
    table = np.full(size, prior, dtype=np.float64)
    table_counts = np.zeros(size, dtype=np.float64)
    np.add.at(table, column, encoded)
    np.add.at(table_counts, column, 1.0)
    table = np.where(table_counts > 0, table / np.maximum(table_counts, 1.0), prior)
    return table.astype(np.float32)


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    x_cat_train = (
        dataset_meta.x_cat['train'] if dataset_meta.x_cat is not None else None
    )
    if x_cat_train is None or x_cat_train.shape[0] == 0:
        return (None, 0, {'cardinalities': cat_cardinalities})
    y_train = np.asarray(dataset_meta.y['train'], dtype=np.float64)
    target_mean = float(y_train.mean()) if y_train.size else 0.0
    target_std = float(y_train.std()) if y_train.size else 1.0
    if target_std == 0.0:
        target_std = 1.0
    prior = target_mean
    lookup_tables: list[Tensor] = []
    for j, c in enumerate(cat_cardinalities):
        size = c + 1
        col = np.asarray(x_cat_train[:, j], dtype=np.int64)
        col = np.clip(col, 0, size - 1)
        table = _compute_oof_lookup(
            column=col,
            y=y_train,
            size=size,
            n_folds=_N_FOLDS,
            smoothing=_SMOOTHING,
            prior=prior,
            seed=j,
        )
        lookup_tables.append(torch.as_tensor(table, dtype=torch.float32))
    embedding = _OOFTargetEncoding(lookup_tables, target_mean, target_std)
    output_dim = len(cat_cardinalities)
    return (embedding.to(device), output_dim, {'cardinalities': cat_cardinalities})
