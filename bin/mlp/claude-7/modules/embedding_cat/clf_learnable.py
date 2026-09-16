# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnableCategoricalEmbedding(nn.Module):
    """Learnable d_embedding per category, one nn.Embedding per column."""

    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding
        bound = d_embedding ** (-0.5)
        self.tables = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=c + 1, embedding_dim=d_embedding)
                for c in cardinalities
            ]
        )
        for table in self.tables:
            nn.init.uniform_(table.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        embeddings = [
            table(x[:, i].clamp(0, table.num_embeddings - 1))
            for i, table in enumerate(self.tables)
        ]
        return torch.cat(embeddings, dim=1)


def build_cat_embedding_v1(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    output_dim = len(cat_cardinalities) * d_embedding
    return (
        LearnableCategoricalEmbedding(cat_cardinalities, d_embedding),
        output_dim,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
