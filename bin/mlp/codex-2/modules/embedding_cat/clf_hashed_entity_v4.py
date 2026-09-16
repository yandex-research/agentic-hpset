# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class HashedEntityEmbeddings(nn.Module):
    def __init__(self, buckets: list[int], d_embedding: int) -> None:
        super().__init__()
        self.buckets = buckets
        self.embeddings = nn.ModuleList(
            [nn.Embedding(bucket, d_embedding) for bucket in buckets]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for i, embedding in enumerate(self.embeddings):
            pieces.append(embedding(torch.remainder(x[:, i], self.buckets[i])))
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v4(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    buckets = [
        max(2, min(cardinality, max(16, min(2048, d_embedding * 32))))
        for cardinality in cat_cardinalities
    ]
    return (
        HashedEntityEmbeddings(buckets, d_embedding),
        len(cat_cardinalities) * d_embedding,
        {
            'cardinalities': cat_cardinalities,
            'buckets': buckets,
            'd_embedding': d_embedding,
        },
    )
