# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class SharedTableCategoricalEmbedding(nn.Module):
    """All categorical columns share one big embedding table. Each column's ids
    are shifted by a column-specific offset so different columns map into
    disjoint slices of the same table. Lets the optimizer share gradient
    information across columns at the cost of giving up per-column structure.
    """

    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        offsets = [0]
        for cardinality in cardinalities:
            offsets.append(offsets[-1] + cardinality + 1)
        total_rows = offsets[-1]
        self.register_buffer('offsets', torch.tensor(offsets[:-1], dtype=torch.long))
        self.table = nn.Embedding(total_rows, d_embedding)
        nn.init.uniform_(
            self.table.weight, -(d_embedding ** (-0.5)), d_embedding ** (-0.5)
        )
        self.d_embedding = d_embedding

    def forward(self, x: Tensor) -> Tensor:
        shifted = x + self.offsets
        out = self.table(shifted)
        return out.flatten(1)


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    embedding = SharedTableCategoricalEmbedding(cat_cardinalities, d_embedding)
    return (
        embedding,
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
