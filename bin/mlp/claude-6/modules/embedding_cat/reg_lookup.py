# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnableCatEmbedding(nn.Module):
    """Independent nn.Embedding per categorical feature; outputs are concatenated."""

    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card + 1, d_embedding) for card in cardinalities]
        )
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding
        bound = d_embedding ** (-0.5)
        for layer in self.embeddings:
            nn.init.uniform_(layer.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        pieces = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
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
    embedding = LearnableCatEmbedding(cat_cardinalities, d_embedding)
    output_dim = len(cat_cardinalities) * d_embedding
    return (embedding, output_dim, {'cardinalities': cat_cardinalities})
