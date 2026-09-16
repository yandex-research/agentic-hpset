# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
import math


class EntityEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], dimensions: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dimension)
                for cardinality, dimension in zip(cardinalities, dimensions)
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)], dim=1
        )


def build_cat_embedding_v1(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    max_dim = int(trial_params['d_embedding'])
    dimensions = [
        min(max_dim, max(2, int(math.ceil(math.sqrt(cardinality + 1)))))
        for cardinality in cat_cardinalities
    ]
    return (
        EntityEmbeddings(cat_cardinalities, dimensions),
        int(sum(dimensions)),
        {'cardinalities': cat_cardinalities, 'dimensions': dimensions},
    )
