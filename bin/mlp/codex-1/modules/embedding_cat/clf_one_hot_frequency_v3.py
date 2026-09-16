# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class OneHotFrequencyEncoding(nn.Module):
    def __init__(self, cardinalities: list[int], frequencies: list[Tensor]) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        for i, frequency in enumerate(frequencies):
            self.register_buffer(f'frequency_{i}', frequency)

    def forward(self, x: Tensor) -> Tensor:
        one_hot = torch.cat(
            [
                F.one_hot(x[:, i], cardinality)
                for i, cardinality in enumerate(self.cardinalities)
            ],
            dim=1,
        ).float()
        frequency_features = torch.stack(
            [
                getattr(self, f'frequency_{i}')[x[:, i]]
                for i in range(len(self.cardinalities))
            ],
            dim=1,
        )
        return torch.cat([one_hot, frequency_features], dim=1)


def _frequency_tables(
    cat_cardinalities: list[int], dataset_meta: Any, device: torch.device
) -> list[Tensor]:
    tables: list[Tensor] = []
    train = dataset_meta.x_cat['train']
    for i, cardinality in enumerate(cat_cardinalities):
        column = torch.as_tensor(train[:, i], dtype=torch.long, device=device).clamp(
            0, cardinality - 1
        )
        counts = torch.bincount(column, minlength=cardinality).float()
        denom = torch.log1p(counts.max()).clamp_min(1.0)
        tables.append(torch.log1p(counts) / denom)
    return tables


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    frequencies = _frequency_tables(cat_cardinalities, dataset_meta, device)
    return (
        OneHotFrequencyEncoding(cat_cardinalities, frequencies),
        sum(cat_cardinalities) + len(cat_cardinalities),
        {'cardinalities': cat_cardinalities},
    )
