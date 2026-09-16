# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch


class FrequencyCategoricalEncoding(nn.Module):
    def __init__(self, encoded_train: np.ndarray, cardinalities: list[int]) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        for i, cardinality in enumerate(cardinalities):
            counts = np.bincount(encoded_train[:, i], minlength=cardinality).astype(
                np.float32
            )
            total = max(float(counts.sum()), 1.0)
            values = np.column_stack(
                [
                    np.log1p(np.r_[counts, 0.0]),
                    np.r_[counts / total, 0.0],
                    np.r_[(counts <= 2).astype(np.float32), 1.0],
                ]
            ).astype(np.float32)
            self.register_buffer(f'values_{i}', torch.as_tensor(values))

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for i, cardinality in enumerate(self.cardinalities):
            values = getattr(self, f'values_{i}')
            pieces.append(values[x[:, i].clamp_max(cardinality)])
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    embedding = FrequencyCategoricalEncoding(
        dataset_meta.x_cat['train'], cat_cardinalities
    )
    return (embedding, len(cat_cardinalities) * 3, {'cardinalities': cat_cardinalities})
