# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np


class FrequencyEncoding(nn.Module):
    def __init__(self, tables: list[Tensor]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding.from_pretrained(table, freeze=True) for table in tables]
        )
        self.output_dim = sum((int(table.shape[1]) for table in tables))

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for column, embedding in enumerate(self.embeddings):
            values = x[:, column].clamp_max(embedding.num_embeddings - 1)
            pieces.append(embedding(values))
        return torch.cat(pieces, dim=1)


class OneHotFrequencyEncoding(nn.Module):
    def __init__(self, cardinalities: list[int], tables: list[Tensor]) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.freq = FrequencyEncoding(tables)
        self.output_dim = sum(cardinalities) + self.freq.output_dim

    def forward(self, x: Tensor) -> Tensor:
        one_hot = torch.cat(
            [
                F.one_hot(x[:, i].clamp_max(cardinality), num_classes=cardinality + 1)[
                    :, :-1
                ]
                for i, cardinality in enumerate(self.cardinalities)
            ],
            dim=1,
        ).float()
        return torch.cat([one_hot, self.freq(x)], dim=1)


def _frequency_tables(
    x_train: np.ndarray, cardinalities: list[int], device: torch.device
) -> list[Tensor]:
    n_rows = max(int(x_train.shape[0]), 1)
    tables: list[Tensor] = []
    for column, cardinality in enumerate(cardinalities):
        values = np.minimum(x_train[:, column], cardinality)
        counts = np.bincount(values, minlength=cardinality + 1).astype(np.float32)
        freq = counts / float(n_rows)
        log_count = np.log1p(counts) / np.log1p(float(n_rows))
        table = np.stack([freq, log_count], axis=1).astype(np.float32)
        tables.append(torch.as_tensor(table, dtype=torch.float32, device=device))
    return tables


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'variant': 'one_hot_frequency', 'cardinalities': []})
    tables = _frequency_tables(dataset_meta.x_cat['train'], cat_cardinalities, device)
    module = OneHotFrequencyEncoding(cat_cardinalities, tables)
    return (
        module,
        module.output_dim,
        {'variant': 'one_hot_frequency', 'cardinalities': cat_cardinalities},
    )
