# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v5``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


def _embedding_dim(cardinality: int, max_dim: int) -> int:
    return min(max_dim, max(2, int(round(cardinality**0.5)) + 1))


class HybridOneHotEntityEmbeddings(nn.Module):
    def __init__(
        self, cardinalities: list[int], max_dim: int, one_hot_limit: int = 16
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.one_hot_limit = one_hot_limit
        self.embedding_columns: list[int] = []
        modules = []
        for column, cardinality in enumerate(cardinalities):
            if cardinality > one_hot_limit:
                self.embedding_columns.append(column)
                modules.append(
                    nn.Embedding(cardinality, _embedding_dim(cardinality, max_dim))
                )
        self.embeddings = nn.ModuleList(modules)
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        embedding_idx = 0
        for column, cardinality in enumerate(self.cardinalities):
            if cardinality <= self.one_hot_limit:
                pieces.append(F.one_hot(x[:, column], cardinality).float())
            else:
                pieces.append(self.embeddings[embedding_idx](x[:, column]))
                embedding_idx += 1
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v5(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    max_dim = int(trial_params['d_embedding'])
    one_hot_limit = 16
    output_dim = sum(
        (
            cardinality
            if cardinality <= one_hot_limit
            else _embedding_dim(cardinality, max_dim)
            for cardinality in cat_cardinalities
        )
    )
    return (
        HybridOneHotEntityEmbeddings(cat_cardinalities, max_dim, one_hot_limit),
        output_dim,
        {'cardinalities': cat_cardinalities, 'one_hot_limit': one_hot_limit},
    )
