# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v5``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnedCategoricalEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality + 1, d_embedding)
                for cardinality in cardinalities
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=d_embedding ** (-0.5))

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [
                embedding(x[:, i].clamp_max(self.cardinalities[i]))
                for i, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )


class CategoryDropoutEmbeddings(LearnedCategoricalEmbeddings):
    def __init__(self, cardinalities: list[int], d_embedding: int, p: float) -> None:
        super().__init__(cardinalities, d_embedding)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0.0:
            columns = []
            for i, cardinality in enumerate(self.cardinalities):
                column = x[:, i].clone()
                mask = torch.rand_like(column.float()) < self.p
                column[mask] = cardinality
                columns.append(column)
            x = torch.stack(columns, dim=1)
        return super().forward(x)


def build_cat_embedding_v5(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    p = min(0.3, float(trial_params.get('dropout', 0.1)))
    return (
        CategoryDropoutEmbeddings(cat_cardinalities, d_embedding, p),
        len(cat_cardinalities) * d_embedding,
        {
            'cardinalities': cat_cardinalities,
            'd_embedding': d_embedding,
            'category_dropout': p,
        },
    )
