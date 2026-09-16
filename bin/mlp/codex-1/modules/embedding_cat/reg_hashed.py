# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import math
import torch.nn as nn
import torch


class HashedCatEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.bucket_sizes = [
            max(8, min(1024, int(math.ceil((cardinality + 1) ** 0.5 * 4.0))))
            for cardinality in cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(bucket_size, d_embedding)
                for bucket_size in self.bucket_sizes
            ]
        )
        self.output_dim = len(cardinalities) * d_embedding

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for i, (bucket_size, embedding) in enumerate(
            zip(self.bucket_sizes, self.embeddings, strict=True)
        ):
            hashed = (x[:, i].clamp_min(0) * 1000003 + i * 97) % bucket_size
            pieces.append(embedding(hashed))
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
    embedding = HashedCatEmbeddings(cat_cardinalities, d_embedding)
    return (
        embedding,
        embedding.output_dim,
        {'cardinalities': cat_cardinalities, 'bucket_sizes': embedding.bucket_sizes},
    )
