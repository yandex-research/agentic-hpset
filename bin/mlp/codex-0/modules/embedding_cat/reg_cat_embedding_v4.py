# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch


class TargetCategoricalEncoding(nn.Module):
    def __init__(
        self,
        encoded_train: np.ndarray,
        y_train: np.ndarray,
        cardinalities: list[int],
        smoothing: float = 20.0,
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        targets = y_train.astype(np.float32)
        global_mean = float(targets.mean())
        for i, cardinality in enumerate(cardinalities):
            codes = encoded_train[:, i]
            counts = np.bincount(codes, minlength=cardinality).astype(np.float32)
            sums = np.bincount(codes, weights=targets, minlength=cardinality).astype(
                np.float32
            )
            encoded = (sums + smoothing * global_mean) / (counts + smoothing)
            values = np.r_[encoded, global_mean].reshape(-1, 1).astype(np.float32)
            self.register_buffer(f'values_{i}', torch.as_tensor(values))

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for i, cardinality in enumerate(self.cardinalities):
            values = getattr(self, f'values_{i}')
            pieces.append(values[x[:, i].clamp_max(cardinality)])
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v4(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    embedding = TargetCategoricalEncoding(
        dataset_meta.x_cat['train'], dataset_meta.y['train'], cat_cardinalities
    )
    return (embedding, len(cat_cardinalities), {'cardinalities': cat_cardinalities})
