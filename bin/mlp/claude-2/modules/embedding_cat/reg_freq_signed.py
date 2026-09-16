# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np


class FreqSignedEmbedding(nn.Module):
    def __init__(
        self, freq_tables: list[Tensor], log_count_tables: list[Tensor]
    ) -> None:
        super().__init__()
        self.n_features = len(freq_tables)
        for i, (f, lc) in enumerate(zip(freq_tables, log_count_tables)):
            self.register_buffer(f'freq_{i}', f)
            self.register_buffer(f'log_count_{i}', lc)

    def forward(self, x: Tensor) -> Tensor:
        outputs: list[Tensor] = []
        for i in range(self.n_features):
            freq = getattr(self, f'freq_{i}')
            log_count = getattr(self, f'log_count_{i}')
            ids = x[:, i].clamp(min=0, max=freq.shape[0] - 1)
            outputs.append(freq[ids].unsqueeze(1))
            outputs.append(log_count[ids].unsqueeze(1))
        return torch.cat(outputs, dim=1)


def build_cat_embedding_v2(
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
    train_size = float(x_cat_train.shape[0])
    freq_tables: list[Tensor] = []
    log_count_tables: list[Tensor] = []
    for j, c in enumerate(cat_cardinalities):
        size = c + 1
        col = x_cat_train[:, j]
        bincount = np.bincount(col.astype(np.int64), minlength=size)
        if bincount.shape[0] < size:
            extra = np.zeros(size - bincount.shape[0], dtype=np.int64)
            bincount = np.concatenate([bincount, extra])
        else:
            bincount = bincount[:size]
        freq = bincount.astype(np.float32) / max(train_size, 1.0)
        log_count = np.log1p(bincount.astype(np.float32))
        freq_tables.append(torch.as_tensor(freq, dtype=torch.float32))
        log_count_tables.append(torch.as_tensor(log_count, dtype=torch.float32))
    embedding = FreqSignedEmbedding(freq_tables, log_count_tables)
    output_dim = len(cat_cardinalities) * 2
    return (embedding.to(device), output_dim, {'cardinalities': cat_cardinalities})
