# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np


class EntityFrequencyEmbeddings(nn.Module):
    def __init__(
        self, cardinalities: list[int], d_embedding: int, frequencies: list[Tensor]
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, d_embedding - 1)
                for cardinality in cardinalities
            ]
        )
        for i, frequency in enumerate(frequencies):
            self.register_buffer(f'frequency_{i}', frequency)
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def _frequency(self, index: int) -> Tensor:
        return getattr(self, f'frequency_{index}')

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, embedding in enumerate(self.embeddings):
            values = x[:, i]
            pieces.append(embedding(values))
            pieces.append(self._frequency(i)[values].unsqueeze(1))
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = max(2, int(trial_params['d_embedding']))
    frequencies: list[Tensor] = []
    x_train = dataset_meta.x_cat['train'] if dataset_meta.x_cat is not None else None
    for column, cardinality in enumerate(cat_cardinalities):
        counts = np.zeros(cardinality, dtype=np.float32)
        if x_train is not None:
            counts += np.bincount(x_train[:, column], minlength=cardinality).astype(
                np.float32
            )[:cardinality]
        frequency = np.log1p(counts) / np.log1p(max(1.0, float(counts.sum())))
        frequencies.append(
            torch.as_tensor(frequency, dtype=torch.float32, device=device)
        )
    return (
        EntityFrequencyEmbeddings(cat_cardinalities, d_embedding, frequencies),
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
