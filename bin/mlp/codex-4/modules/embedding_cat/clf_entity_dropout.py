# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class EntityDropoutEmbeddings(nn.Module):
    def __init__(
        self, cardinalities: list[int], d_embedding: int, dropout: float
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_embedding) for cardinality in cardinalities]
        )
        self.dropout = float(min(max(dropout, 0.0), 0.35))
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        pieces = [
            embedding(x[:, index]) for index, embedding in enumerate(self.embeddings)
        ]
        stacked = torch.stack(pieces, dim=1)
        if self.training and self.dropout > 0.0:
            keep = torch.empty(
                stacked.shape[0], stacked.shape[1], 1, device=stacked.device
            ).bernoulli_(1.0 - self.dropout)
            stacked = stacked * keep / (1.0 - self.dropout)
        return stacked.flatten(1)


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    dropout = float(trial_params.get('dropout', 0.0)) * 0.5
    return (
        EntityDropoutEmbeddings(cat_cardinalities, d_embedding, dropout),
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
