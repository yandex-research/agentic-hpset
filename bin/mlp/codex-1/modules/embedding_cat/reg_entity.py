# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import math
import torch.nn as nn
import torch


def _embedding_dim(cardinality: int, d_embedding: int) -> int:
    return max(2, min(d_embedding, int(math.ceil(cardinality**0.5)) + 1))


class EntityEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        dims = [
            _embedding_dim(cardinality, d_embedding) for cardinality in cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality + 1, dim, padding_idx=cardinality)
                for cardinality, dim in zip(cardinalities, dims, strict=True)
            ]
        )
        self.output_dim = sum(dims)

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for i, (cardinality, embedding) in enumerate(
            zip(self.cardinalities, self.embeddings, strict=True)
        ):
            values = x[:, i].clamp(0, cardinality)
            pieces.append(embedding(values))
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v1(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    embedding = EntityEmbeddings(cat_cardinalities, int(trial_params['d_embedding']))
    return (embedding, embedding.output_dim, {'cardinalities': cat_cardinalities})
