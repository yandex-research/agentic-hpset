# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v5``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class HybridOneHotEntityEncoding(nn.Module):
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_embedding) for cardinality in cardinalities]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=d_embedding ** (-0.5))

    def forward(self, x: Tensor) -> Tensor:
        one_hot = torch.cat(
            [
                F.one_hot(x[:, i], cardinality)
                for i, cardinality in enumerate(self.cardinalities)
            ],
            dim=1,
        ).float()
        dense = torch.cat(
            [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)], dim=1
        )
        return torch.cat([one_hot, dense], dim=1)


def build_cat_embedding_v5(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = max(2, int(trial_params['d_embedding']) // 2)
    output_dim = sum(cat_cardinalities) + len(cat_cardinalities) * d_embedding
    return (
        HybridOneHotEntityEncoding(cat_cardinalities, d_embedding),
        output_dim,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
