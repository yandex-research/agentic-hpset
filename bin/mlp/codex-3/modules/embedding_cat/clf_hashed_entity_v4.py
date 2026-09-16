# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


def _bucket_count(cardinality: int) -> int:
    return min(cardinality, max(8, int(round(4.0 * cardinality**0.5))))


def _embedding_dim(bucket_count: int, max_dim: int) -> int:
    return min(max_dim, max(2, int(round(bucket_count**0.5)) + 1))


class HashedEntityEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], max_dim: int) -> None:
        super().__init__()
        self.bucket_counts = [
            _bucket_count(cardinality) for cardinality in cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(bucket_count, _embedding_dim(bucket_count, max_dim))
                for bucket_count in self.bucket_counts
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for i, embedding in enumerate(self.embeddings):
            pieces.append(embedding(torch.remainder(x[:, i], self.bucket_counts[i])))
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v4(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    max_dim = int(trial_params['d_embedding'])
    bucket_counts = [_bucket_count(cardinality) for cardinality in cat_cardinalities]
    output_dim = sum(
        (_embedding_dim(bucket_count, max_dim) for bucket_count in bucket_counts)
    )
    return (
        HashedEntityEmbeddings(cat_cardinalities, max_dim),
        output_dim,
        {'cardinalities': cat_cardinalities, 'bucket_counts': bucket_counts},
    )
