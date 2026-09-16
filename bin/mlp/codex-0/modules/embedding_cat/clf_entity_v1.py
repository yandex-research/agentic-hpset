# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class EntityEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_embedding) for cardinality in cardinalities]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=d_embedding ** (-0.5))

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
    d_embedding = int(trial_params['d_embedding'])
    return (
        EntityEmbeddings(cat_cardinalities, d_embedding),
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
