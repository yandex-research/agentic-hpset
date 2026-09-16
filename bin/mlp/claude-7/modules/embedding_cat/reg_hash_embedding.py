# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2_hash``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class HashEmbedding(nn.Module):
    """Hash trick: each categorical column is hashed into one of `n_buckets`,
    then mapped through a shared embedding table. Memory cost is independent
    of cardinality. Useful for very high-cardinality categorical features.
    """

    def __init__(self, n_columns: int, d_embedding: int, n_buckets: int) -> None:
        super().__init__()
        self.n_columns = n_columns
        self.n_buckets = n_buckets
        self.column_offsets = nn.Parameter(
            torch.randint(0, max(n_buckets, 1), (n_columns,)), requires_grad=False
        )
        self.embedding = nn.Embedding(n_buckets * n_columns, d_embedding)
        nn.init.uniform_(
            self.embedding.weight, -(d_embedding ** (-0.5)), d_embedding ** (-0.5)
        )

    def forward(self, x: Tensor) -> Tensor:
        offsets = self.column_offsets.unsqueeze(0)
        hashed = (x + offsets) % self.n_buckets
        ids = hashed + torch.arange(self.n_columns, device=x.device).mul(
            self.n_buckets
        ).unsqueeze(0)
        return self.embedding(ids).flatten(1)


def build_cat_embedding_v2_hash(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    avg_card = max(int(sum(cat_cardinalities) / max(len(cat_cardinalities), 1)), 1)
    n_buckets = max(min(avg_card, 64), 8)
    embedding = HashEmbedding(len(cat_cardinalities), d_embedding, n_buckets)
    return (
        embedding,
        d_embedding * len(cat_cardinalities),
        {'n_buckets': n_buckets, 'd_embedding': d_embedding},
    )
