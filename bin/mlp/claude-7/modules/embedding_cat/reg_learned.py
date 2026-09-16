# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1_learned``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnedCategoricalEmbeddings(nn.Module):
    """Per-column learned embeddings; concatenates over the column axis.

    Each categorical column gets its own ``nn.Embedding(card+1, d)`` table
    (the +1 leaves a slot for unknown values). Output is flattened to
    (B, n_columns * d_embedding).
    """

    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card + 1, d_embedding) for card in cardinalities]
        )
        bound = d_embedding ** (-0.5)
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -bound, bound)
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding

    def forward(self, x: Tensor) -> Tensor:
        pieces = [
            emb(x[:, i].clamp(0, emb.num_embeddings - 1))
            for i, emb in enumerate(self.embeddings)
        ]
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v1_learned(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    embedding = LearnedCategoricalEmbeddings(cat_cardinalities, d_embedding)
    return (
        embedding,
        d_embedding * len(cat_cardinalities),
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
