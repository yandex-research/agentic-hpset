# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_hash``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class HashEmbedding(nn.Module):
    def __init__(
        self, cardinalities: list[int], d_embedding: int, n_buckets: int
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.n_buckets = n_buckets
        self.d_embedding = d_embedding
        self.embedding = nn.Embedding(n_buckets * len(cardinalities), d_embedding)
        nn.init.normal_(self.embedding.weight, std=d_embedding ** (-0.5))
        salts = torch.tensor(
            [
                1469598103934665603 + i * 1099511628211
                for i in range(len(cardinalities))
            ],
            dtype=torch.long,
        )
        self.register_buffer('salts', salts)

    def forward(self, x: Tensor) -> Tensor:
        hashed = x.long() * self.salts % self.n_buckets
        offset = torch.arange(len(self.cardinalities), device=x.device) * self.n_buckets
        bucket_ids = hashed + offset
        embedded = self.embedding(bucket_ids)
        return embedded.flatten(1)


def build_cat_embedding_hash(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    n_buckets = max(8, min(max(cat_cardinalities) + 1, 4096))
    embedding = HashEmbedding(cat_cardinalities, d_embedding, n_buckets)
    output_dim = len(cat_cardinalities) * d_embedding
    return (
        embedding,
        output_dim,
        {'cardinalities': cat_cardinalities, 'n_buckets': n_buckets},
    )
