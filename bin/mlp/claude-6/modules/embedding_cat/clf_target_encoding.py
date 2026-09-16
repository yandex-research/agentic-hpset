# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class TargetEncoding(nn.Module):
    def __init__(self, lookups: list[Tensor], cardinalities: list[int]) -> None:
        super().__init__()
        for i, table in enumerate(lookups):
            self.register_buffer(f'lookup_{i}', table)
        self.n_features = len(lookups)
        self.out_per_feat = lookups[0].shape[1] if lookups else 0
        self.cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for i in range(self.n_features):
            table: Tensor = getattr(self, f'lookup_{i}')
            idx = x[:, i].clamp(0, table.shape[0] - 1)
            outs.append(F.one_hot(idx, self.cardinalities[i]).float())
            outs.append(table.index_select(0, idx))
        return torch.cat(outs, dim=1)


def _smoothed_target_means(
    y_train: np.ndarray,
    cat_column: np.ndarray,
    cardinality: int,
    n_classes: int,
    is_binclass: bool,
    smoothing: float | None = None,
) -> np.ndarray:
    prior_strength = float(
        max(20.0, np.sqrt(max(y_train.shape[0], 1))) if smoothing is None else smoothing
    )
    if is_binclass:
        out_dim = 1
        global_mean = float(y_train.mean())
        table = np.full((cardinality, out_dim), global_mean, dtype=np.float32)
        for c in range(cardinality):
            mask = cat_column == c
            count = int(mask.sum())
            if count == 0:
                continue
            mean = float(y_train[mask].mean())
            effective_count = max(count - 1, 0)
            smoothed = (effective_count * mean + prior_strength * global_mean) / (
                effective_count + prior_strength
            )
            table[c, 0] = smoothed
        return table
    out_dim = n_classes
    global_freq = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    global_freq /= max(global_freq.sum(), 1.0)
    table = np.tile(global_freq.astype(np.float32), (cardinality, 1))
    for c in range(cardinality):
        mask = cat_column == c
        count = int(mask.sum())
        if count == 0:
            continue
        local = np.bincount(y_train[mask], minlength=n_classes).astype(np.float64)
        local /= max(local.sum(), 1.0)
        effective_count = max(count - 1, 0)
        smoothed = (effective_count * local + prior_strength * global_freq) / (
            effective_count + prior_strength
        )
        table[c] = smoothed.astype(np.float32)
    return table


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    y_train = dataset_meta.y['train']
    x_cat_train = dataset_meta.x_cat['train']
    out_dim = 1 if dataset_meta.is_binclass else int(dataset_meta.n_classes)
    lookups: list[Tensor] = []
    for i, card in enumerate(cat_cardinalities):
        table = _smoothed_target_means(
            y_train,
            x_cat_train[:, i],
            card,
            int(dataset_meta.n_classes),
            dataset_meta.is_binclass,
        )
        lookups.append(torch.as_tensor(table, dtype=torch.float32, device=device))
    module = TargetEncoding(lookups, cat_cardinalities)
    return (
        module,
        sum(cat_cardinalities) + len(cat_cardinalities) * out_dim,
        {'cardinalities': cat_cardinalities},
    )
