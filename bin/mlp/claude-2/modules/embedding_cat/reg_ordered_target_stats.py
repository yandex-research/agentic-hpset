# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np


class _OrderedTargetStats(nn.Module):
    def __init__(self, lookup_tables: list[Tensor]) -> None:
        super().__init__()
        self.n_features = len(lookup_tables)
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
_N_PERMUTATIONS = 4


def _ordered_table(
    column: np.ndarray,
    y: np.ndarray,
    size: int,
    smoothing: float,
    prior: float,
    seed: int,
) -> np.ndarray:
    n = column.shape[0]
    if n == 0:
        return np.full(size, prior, dtype=np.float32)
    accum_sum = np.zeros(size, dtype=np.float64)
    accum_count = np.zeros(size, dtype=np.float64)
    final_table = np.full(size, prior, dtype=np.float64)
    final_counts = np.zeros(size, dtype=np.float64)
    for perm_seed in range(_N_PERMUTATIONS):
        rng = np.random.RandomState(seed * 31 + perm_seed)
        perm = rng.permutation(n)
        col_perm = column[perm]
        y_perm = y[perm]
        local_sum = np.zeros(size, dtype=np.float64)
        local_count = np.zeros(size, dtype=np.float64)
        encoded = np.zeros(n, dtype=np.float64)
        for i in range(n):
            cat = col_perm[i]
            cnt = local_count[cat]
            if cnt > 0:
                encoded[i] = (local_sum[cat] + smoothing * prior) / (cnt + smoothing)
            else:
                encoded[i] = prior
            local_sum[cat] += y_perm[i]
            local_count[cat] += 1.0
        np.add.at(final_table, col_perm, encoded)
        np.add.at(final_counts, col_perm, 1.0)
    final_table = np.where(
        final_counts > 0, final_table / np.maximum(final_counts, 1.0), prior
    )
    return final_table.astype(np.float32)


def build_cat_embedding_v4(
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
    prior = float(y_train.mean()) if y_train.size else 0.0
    lookup_tables: list[Tensor] = []
    for j, c in enumerate(cat_cardinalities):
        size = c + 1
        col = np.asarray(x_cat_train[:, j], dtype=np.int64)
        col = np.clip(col, 0, size - 1)
        table = _ordered_table(
            column=col, y=y_train, size=size, smoothing=_SMOOTHING, prior=prior, seed=j
        )
        lookup_tables.append(torch.as_tensor(table, dtype=torch.float32))
    embedding = _OrderedTargetStats(lookup_tables)
    output_dim = len(cat_cardinalities)
    return (embedding.to(device), output_dim, {'cardinalities': cat_cardinalities})
