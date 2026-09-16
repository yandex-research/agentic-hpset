# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np


class TargetStatsEmbeddings(nn.Module):
    def __init__(
        self, cardinalities: list[int], d_embedding: int, stats: list[Tensor]
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_embedding) for cardinality in cardinalities]
        )
        for i, values in enumerate(stats):
            self.register_buffer(f'stats_{i}', values)
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def _stats(self, index: int) -> Tensor:
        return getattr(self, f'stats_{index}')

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, embedding in enumerate(self.embeddings):
            values = x[:, i]
            pieces.append(embedding(values))
            pieces.append(self._stats(i)[values])
        return torch.cat(pieces, dim=1)


def _binary_stats(values: np.ndarray, y: np.ndarray, cardinality: int) -> np.ndarray:
    prior = float(y.mean()) if y.size else 0.5
    counts = np.bincount(values, minlength=cardinality).astype(np.float32)
    positives = np.bincount(
        values, weights=y.astype(np.float32), minlength=cardinality
    ).astype(np.float32)
    rate = (positives + 5.0 * prior) / (counts + 5.0)
    return rate[:, None].astype(np.float32)


def _multiclass_stats(
    values: np.ndarray, y: np.ndarray, cardinality: int, n_classes: int
) -> np.ndarray:
    prior = np.bincount(y, minlength=n_classes).astype(np.float32)
    prior = prior / max(1.0, float(prior.sum()))
    counts = np.bincount(values, minlength=cardinality).astype(np.float32)
    stats = np.zeros((cardinality, n_classes), dtype=np.float32)
    for cls in range(n_classes):
        cls_counts = np.bincount(
            values, weights=(y == cls).astype(np.float32), minlength=cardinality
        ).astype(np.float32)
        stats[:, cls] = (cls_counts + 5.0 * prior[cls]) / (counts + 5.0)
    return stats


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    x_train = dataset_meta.x_cat['train'] if dataset_meta.x_cat is not None else None
    y_train = dataset_meta.y['train'].astype(np.int64)
    stats: list[Tensor] = []
    stat_dim = 1 if dataset_meta.is_binclass else int(dataset_meta.n_classes)
    for column, cardinality in enumerate(cat_cardinalities):
        if x_train is None:
            array = np.zeros((cardinality, stat_dim), dtype=np.float32)
        elif dataset_meta.is_binclass:
            array = _binary_stats(x_train[:, column], y_train, cardinality)
        else:
            array = _multiclass_stats(
                x_train[:, column], y_train, cardinality, stat_dim
            )
        stats.append(torch.as_tensor(array, dtype=torch.float32, device=device))
    output_dim = len(cat_cardinalities) * (d_embedding + stat_dim)
    return (
        TargetStatsEmbeddings(cat_cardinalities, d_embedding, stats),
        output_dim,
        {
            'cardinalities': cat_cardinalities,
            'd_embedding': d_embedding,
            'stat_dim': stat_dim,
        },
    )
