# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnedDenseCategoricalEmbedding(nn.Module):
    """Per-feature learnable dense categorical embedding.

    Each categorical feature gets its own embedding table mapping cardinality -> d_embedding.
    Outputs concatenated per-feature dense vectors.
    """

    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality + 1, d_embedding)
                for cardinality in cardinalities
            ]
        )
        for emb in self.embeddings:
            nn.init.uniform_(
                emb.weight, -(d_embedding ** (-0.5)), d_embedding ** (-0.5)
            )

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for i, emb in enumerate(self.embeddings):
            idx = x[:, i].clamp(0, emb.num_embeddings - 1)
            pieces.append(emb(idx))
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v1(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    return (
        LearnedDenseCategoricalEmbedding(cat_cardinalities, d_embedding),
        d_embedding * len(cat_cardinalities),
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
