# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnedCategoricalEmbedding(nn.Module):
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, d_embedding) for c in cardinalities]
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=d_embedding ** (-0.5))
        self.d_embedding = d_embedding

    def forward(self, x: Tensor) -> Tensor:
        outs = [self.embeddings[i](x[:, i]) for i in range(len(self.cardinalities))]
        return torch.cat(outs, dim=1)


def build_cat_embedding_v1(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    module = LearnedCategoricalEmbedding(cat_cardinalities, d_embedding)
    output_dim = len(cat_cardinalities) * d_embedding
    return (
        module,
        output_dim,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
