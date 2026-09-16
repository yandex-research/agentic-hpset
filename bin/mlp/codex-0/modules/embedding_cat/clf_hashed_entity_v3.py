# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
import math


class HashedEntityEmbeddings(nn.Module):
    def __init__(
        self, cardinalities: list[int], bucket_sizes: list[int], d_embedding: int
    ) -> None:
        super().__init__()
        self.bucket_sizes = bucket_sizes
        self.embeddings = nn.ModuleList(
            [nn.Embedding(bucket_size, d_embedding) for bucket_size in bucket_sizes]
        )
        self.offsets = nn.Parameter(torch.empty(len(cardinalities), d_embedding))
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=d_embedding ** (-0.5))
        nn.init.normal_(self.offsets, mean=0.0, std=d_embedding ** (-0.5))

    def forward(self, x: Tensor) -> Tensor:
        pieces = [
            embedding(torch.remainder(x[:, i], bucket_size)) + self.offsets[i]
            for i, (embedding, bucket_size) in enumerate(
                zip(self.embeddings, self.bucket_sizes)
            )
        ]
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    bucket_sizes = [
        min(cardinality, max(2, int(math.ceil(math.sqrt(cardinality) * 4))))
        for cardinality in cat_cardinalities
    ]
    return (
        HashedEntityEmbeddings(cat_cardinalities, bucket_sizes, d_embedding),
        len(cat_cardinalities) * d_embedding,
        {
            'cardinalities': cat_cardinalities,
            'bucket_sizes': bucket_sizes,
            'd_embedding': d_embedding,
        },
    )
