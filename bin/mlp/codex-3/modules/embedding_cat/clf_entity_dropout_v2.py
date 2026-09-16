# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


def _embedding_dim(cardinality: int, max_dim: int) -> int:
    return min(max_dim, max(2, int(round(cardinality**0.5)) + 1))


class EntityDropoutEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], max_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, _embedding_dim(cardinality, max_dim))
                for cardinality in cardinalities
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        pieces = [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]
        out = torch.cat(pieces, dim=1)
        return nn.functional.dropout(out, p=self.dropout, training=self.training)


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    max_dim = int(trial_params['d_embedding'])
    dropout = min(0.35, max(0.05, float(trial_params['dropout']) * 0.5))
    output_dim = sum(
        (_embedding_dim(cardinality, max_dim) for cardinality in cat_cardinalities)
    )
    return (
        EntityDropoutEmbeddings(cat_cardinalities, max_dim, dropout),
        output_dim,
        {'cardinalities': cat_cardinalities, 'max_dim': max_dim, 'dropout': dropout},
    )
