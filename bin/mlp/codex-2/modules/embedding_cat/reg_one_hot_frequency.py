# ruff: noqa
"""Standalone implementation for ``build_one_hot_frequency``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class FrequencyWeightedOneHot(nn.Module):
    def __init__(
        self, cardinalities: list[int], train_values: dict[str, Tensor] | None = None
    ) -> None:
        super().__init__()
        weights: list[Tensor] = []
        x_train = None if train_values is None else train_values.get('train')
        for feature, cardinality in enumerate(cardinalities):
            weight = torch.ones(cardinality, dtype=torch.float32)
            if x_train is not None and cardinality > 0:
                train_column = torch.as_tensor(x_train[:, feature], dtype=torch.long)
                counts = torch.bincount(
                    train_column.clamp_min(0).clamp_max(cardinality - 1),
                    minlength=cardinality,
                ).float()
                inv = counts.sum().clamp_min(1.0).sqrt() / counts.clamp_min(1.0).sqrt()
                weight = inv / inv.mean().clamp_min(1e-06)
            weights.append(weight)
        self.cardinalities = cardinalities
        for idx, weight in enumerate(weights):
            self.register_buffer(f'weight_{idx}', weight)

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for idx, cardinality in enumerate(self.cardinalities):
            encoded = F.one_hot(x[:, idx].clamp_max(cardinality), cardinality + 1)[
                :, :-1
            ].float()
            pieces.append(encoded * getattr(self, f'weight_{idx}')[None])
        return torch.cat(pieces, dim=1)


def build_one_hot_frequency(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'variant': 'one_hot_frequency', 'cardinalities': []})
    module = FrequencyWeightedOneHot(cat_cardinalities, dataset_meta.x_cat)
    return (
        module,
        sum(cat_cardinalities),
        {'variant': 'one_hot_frequency', 'cardinalities': cat_cardinalities},
    )
