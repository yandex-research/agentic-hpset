# ruff: noqa
"""Standalone implementation for ``build_hashed_embeddings``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class HashedEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], dims: list[int]) -> None:
        super().__init__()
        self.bucket_counts = [
            max(4, min(cardinality + 1, int(2 * (cardinality + 1) ** 0.5) + 8))
            for cardinality in cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(bucket_count, dim)
                for bucket_count, dim in zip(self.bucket_counts, dims)
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=embedding.embedding_dim ** (-0.5))

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for idx, (bucket_count, embedding) in enumerate(
            zip(self.bucket_counts, self.embeddings)
        ):
            hashed = (x[:, idx] * 1315423911 + idx * 2654435761).remainder(bucket_count)
            pieces.append(embedding(hashed))
        return torch.cat(pieces, dim=1)


def _embedding_dims(cardinalities: list[int], d_embedding: int) -> list[int]:
    dims: list[int] = []
    for cardinality in cardinalities:
        suggested = max(2, min(d_embedding, int(round((cardinality + 1) ** 0.5)) + 1))
        dims.append(suggested)
    return dims


def build_hashed_embeddings(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'variant': 'hashed', 'cardinalities': []})
    dims = _embedding_dims(cat_cardinalities, int(trial_params['d_embedding']))
    module = HashedEmbeddings(cat_cardinalities, dims)
    return (
        module,
        sum(dims),
        {
            'variant': 'hashed',
            'cardinalities': cat_cardinalities,
            'bucket_counts': module.bucket_counts,
            'dims': dims,
        },
    )
