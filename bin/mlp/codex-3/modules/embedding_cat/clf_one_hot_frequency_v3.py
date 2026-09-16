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
        for i, freq in enumerate(frequencies):
            self.register_buffer(f'frequency_{i}', freq)

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, cardinality in enumerate(self.cardinalities):
            pieces.append(F.one_hot(x[:, i], cardinality).float())
            pieces.append(getattr(self, f'frequency_{i}')[x[:, i]].unsqueeze(1))
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    frequencies: list[Tensor] = []
    x_train = dataset_meta.x_cat['train']
    for column, cardinality in enumerate(cat_cardinalities):
        counts = torch.bincount(
            torch.as_tensor(x_train[:, column], dtype=torch.long), minlength=cardinality
        ).float()
        freq = torch.log1p(counts) / torch.log1p(counts.max().clamp_min(1.0))
        frequencies.append(freq.to(device))
    return (
        OneHotFrequencyEncoding(cat_cardinalities, frequencies),
        sum(cat_cardinalities) + len(cat_cardinalities),
        {'cardinalities': cat_cardinalities},
    )
