# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import math
import torch.nn as nn
import torch


class EntityEmbedding(nn.Module):
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding
        self.offsets: list[int] = []
        running = 0
        for c in cardinalities:
            self.offsets.append(running)
            running += int(c)
        self.total = running
        self.embedding = nn.Embedding(running, d_embedding)
        nn.init.normal_(
            self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_embedding)
        )
        self.register_buffer(
            'offsets_tensor', torch.tensor(self.offsets, dtype=torch.long)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_off = x + self.offsets_tensor
        emb = self.embedding(x_off)
        return emb.flatten(1)


def build_cat_embedding_v1(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    module = EntityEmbedding(cat_cardinalities, d_embedding)
    return (
        module,
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
