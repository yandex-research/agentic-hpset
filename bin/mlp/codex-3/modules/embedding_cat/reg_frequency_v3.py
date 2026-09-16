# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np


class FrequencyEncoding(nn.Module):
    def __init__(self, cardinalities: list[int], frequencies: Tensor) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.register_buffer('frequencies', frequencies)

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, cardinality in enumerate(self.cardinalities):
            values = torch.where(
                x[:, i] < cardinality, x[:, i], torch.full_like(x[:, i], cardinality)
            )
            pieces.append(self.frequencies[i, values].unsqueeze(1))
        return torch.cat(pieces, dim=1)


def _compute_frequencies(cat_cardinalities: list[int], dataset_meta: Any) -> Tensor:
    max_cardinality = max(cat_cardinalities) + 1
    frequencies = np.zeros((len(cat_cardinalities), max_cardinality), dtype=np.float32)
    train = dataset_meta.x_cat['train']
    denominator = np.log1p(max(train.shape[0], 1))
    for i, cardinality in enumerate(cat_cardinalities):
        counts = np.bincount(train[:, i], minlength=cardinality).astype(np.float32)
        frequencies[i, :cardinality] = np.log1p(counts) / denominator
        frequencies[i, cardinality] = 0.0
    return torch.as_tensor(frequencies, dtype=torch.float32)


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    frequencies = _compute_frequencies(cat_cardinalities, dataset_meta).to(device)
    return (
        FrequencyEncoding(cat_cardinalities, frequencies),
        len(cat_cardinalities),
        {'cardinalities': cat_cardinalities, 'frequencies': frequencies},
    )
