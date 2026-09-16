# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnedCatEmbedding(nn.Module):
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_embedding) for card in cardinalities]
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=d_embedding ** (-0.5))

    def forward(self, x: Tensor) -> Tensor:
        parts = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=1)


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
        LearnedCatEmbedding(cat_cardinalities, d_embedding),
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
