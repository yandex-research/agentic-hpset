# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class HashingCategoricalEmbedding(nn.Module):
    """Hashes each categorical id into a shared, fixed-size embedding table per
    column. Useful when high-cardinality categoricals would blow up the
    parameter count of a per-category lookup table.

    For each column we keep a separate small table of size `n_buckets`. The
    forward maps the integer id to a bucket via modulo and looks up the row.
    """

    def __init__(
        self, cardinalities: list[int], d_embedding: int, n_buckets: int
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding
        self.n_buckets = n_buckets
        self.tables = nn.ModuleList(
            [nn.Embedding(n_buckets, d_embedding) for _ in cardinalities]
        )
        bound = d_embedding ** (-0.5)
        for table in self.tables:
            nn.init.uniform_(table.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        outs: list[Tensor] = []
        for i, table in enumerate(self.tables):
            ids = (x[:, i] % self.n_buckets).clamp_min_(0)
            outs.append(table(ids))
        return torch.cat(outs, dim=1)


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    n_buckets = min(256, max((int(c) + 1 for c in cat_cardinalities)))
    embedding = HashingCategoricalEmbedding(cat_cardinalities, d_embedding, n_buckets)
    return (
        embedding,
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'n_buckets': n_buckets},
    )
