# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnedCategoricalEmbedding(nn.Module):
    """Per-column learned embedding tables. Each table has rows
    `cardinality + 1` so unknown categories that the preprocess stage maps to
    `train_max + 1` get a dedicated trainable vector. Output is concatenated
    across columns to match the (B, total_dim) layout the MLP expects."""

    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding
        self.tables = nn.ModuleList(
            [
                nn.Embedding(cardinality + 1, d_embedding)
                for cardinality in cardinalities
            ]
        )
        bound = d_embedding ** (-0.5)
        for table in self.tables:
            nn.init.uniform_(table.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([table(x[:, i]) for i, table in enumerate(self.tables)], dim=1)


def build_cat_embedding_v1(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    embedding = LearnedCategoricalEmbedding(cat_cardinalities, d_embedding)
    return (
        embedding,
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
