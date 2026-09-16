# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np


class FrequencyAwareEmbedding(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        d_embedding: int,
        shrinkage_per_feature: list[Tensor],
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_embedding) for card in cardinalities]
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=d_embedding ** (-0.5))
        for i, shrink in enumerate(shrinkage_per_feature):
            self.register_buffer(f'shrink_{i}', shrink)

    def forward(self, x: Tensor) -> Tensor:
        parts = []
        for i, emb in enumerate(self.embeddings):
            shrink: Tensor = getattr(self, f'shrink_{i}')
            idx = x[:, i].clamp(0, shrink.shape[0] - 1)
            scale = shrink.index_select(0, idx).unsqueeze(1)
            parts.append(emb(idx) * scale)
        return torch.cat(parts, dim=1)


def _frequency_shrinkage(
    cat_column: np.ndarray, cardinality: int, k: float = 5.0
) -> np.ndarray:
    counts = np.bincount(cat_column, minlength=cardinality).astype(np.float32)
    return counts / (counts + k)


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    x_cat_train = dataset_meta.x_cat['train']
    shrinkage = []
    for i, card in enumerate(cat_cardinalities):
        shrink = _frequency_shrinkage(x_cat_train[:, i], card)
        shrinkage.append(torch.as_tensor(shrink, dtype=torch.float32, device=device))
    module = FrequencyAwareEmbedding(cat_cardinalities, d_embedding, shrinkage)
    return (
        module,
        len(cat_cardinalities) * d_embedding,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
