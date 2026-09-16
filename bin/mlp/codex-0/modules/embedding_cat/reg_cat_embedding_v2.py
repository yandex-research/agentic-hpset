# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LowRankCategoricalEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], rank: int, d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, rank) for cardinality in cardinalities]
        )
        self.projections = nn.ModuleList(
            [nn.Linear(rank, d_embedding) for _ in cardinalities]
        )

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for i, (embedding, projection) in enumerate(
            zip(self.embeddings, self.projections)
        ):
            pieces.append(
                projection(embedding(x[:, i].clamp_max(self.cardinalities[i])))
            )
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    rank = max(2, min(8, d_embedding // 2))
    return (
        LowRankCategoricalEmbeddings(cat_cardinalities, rank, d_embedding),
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'rank': rank, 'd_embedding': d_embedding},
    )
