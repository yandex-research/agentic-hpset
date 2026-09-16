# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

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
        for i, values in enumerate(frequencies):
            self.register_buffer(f'frequency_{i}', values)

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, cardinality in enumerate(self.cardinalities):
            values = x[:, i].clamp(0, cardinality)
            pieces.append(F.one_hot(values, cardinality + 1)[:, :-1].float())
            frequencies = getattr(self, f'frequency_{i}')
            pieces.append(frequencies[values].unsqueeze(1))
        return torch.cat(pieces, dim=1)


def _compute_frequencies(train: Tensor, cardinalities: list[int]) -> list[Tensor]:
    frequencies = []
    for i, cardinality in enumerate(cardinalities):
        values = train[:, i].clamp(0, cardinality)
        counts = torch.bincount(values, minlength=cardinality + 1).float()
        counts[-1] = 0.0
        total = counts[:-1].sum().clamp_min(1.0)
        frequencies.append(torch.log1p(counts / total))
    return frequencies


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    train = torch.as_tensor(dataset_meta.x_cat['train'], dtype=torch.long)
    frequencies = _compute_frequencies(train, cat_cardinalities)
    return (
        OneHotFrequencyEncoding(cat_cardinalities, frequencies),
        sum(cat_cardinalities) + len(cat_cardinalities),
        {'cardinalities': cat_cardinalities},
    )
