# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v4``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class WeightedOneHotEncoding(nn.Module):
    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        output_dim = sum(cardinalities)
        self.weight = nn.Parameter(torch.ones(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: Tensor) -> Tensor:
        encoded = torch.cat(
            [
                F.one_hot(x[:, i], cardinality)
                for i, cardinality in enumerate(self.cardinalities)
            ],
            dim=1,
        ).float()
        return encoded * self.weight + self.bias


def build_cat_embedding_v4(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    return (
        WeightedOneHotEncoding(cat_cardinalities),
        sum(cat_cardinalities),
        {'cardinalities': cat_cardinalities},
    )
