# ruff: noqa
"""Standalone implementation for ``build_entity_embeddings``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class EntityEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], dims: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality + 1, dim)
                for cardinality, dim in zip(cardinalities, dims)
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=embedding.embedding_dim ** (-0.5))

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [
                embedding(x[:, idx].clamp_max(embedding.num_embeddings - 1))
                for idx, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )


def _embedding_dims(cardinalities: list[int], d_embedding: int) -> list[int]:
    dims: list[int] = []
    for cardinality in cardinalities:
        suggested = max(2, min(d_embedding, int(round((cardinality + 1) ** 0.5)) + 1))
        dims.append(suggested)
    return dims


def build_entity_embeddings(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'variant': 'entity', 'cardinalities': []})
    dims = _embedding_dims(cat_cardinalities, int(trial_params['d_embedding']))
    return (
        EntityEmbeddings(cat_cardinalities, dims),
        sum(dims),
        {'variant': 'entity', 'cardinalities': cat_cardinalities, 'dims': dims},
    )
