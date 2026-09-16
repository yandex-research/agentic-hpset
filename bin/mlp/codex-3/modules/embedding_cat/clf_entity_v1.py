# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


def _embedding_dim(cardinality: int, max_dim: int) -> int:
    return min(max_dim, max(2, int(round(cardinality**0.5)) + 1))


class EntityEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], max_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, _embedding_dim(cardinality, max_dim))
                for cardinality in cardinalities
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)

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
    output_dim = sum(
        (_embedding_dim(cardinality, max_dim) for cardinality in cat_cardinalities)
    )
    return (
        EntityEmbeddings(cat_cardinalities, max_dim),
        output_dim,
        {'cardinalities': cat_cardinalities, 'max_dim': max_dim},
    )
