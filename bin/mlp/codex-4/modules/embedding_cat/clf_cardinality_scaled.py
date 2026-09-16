# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class CardinalityScaledEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], dims: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dim)
                for cardinality, dim in zip(cardinalities, dims)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in dims])
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        pieces = [
            norm(embedding(x[:, index]))
            for index, (embedding, norm) in enumerate(zip(self.embeddings, self.norms))
        ]
        return torch.cat(pieces, dim=1)


def _scaled_dim(cardinality: int, max_dim: int) -> int:
    return max(2, min(max_dim, int(round(1.6 * cardinality**0.5)) + 1))


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    max_dim = int(trial_params['d_embedding'])
    dims = [_scaled_dim(cardinality, max_dim) for cardinality in cat_cardinalities]
    return (
        CardinalityScaledEmbeddings(cat_cardinalities, dims),
        sum(dims),
        {'cardinalities': cat_cardinalities, 'dims': dims},
    )
