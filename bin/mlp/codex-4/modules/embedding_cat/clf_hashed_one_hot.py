# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v4``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class HashedOneHotEncoding(nn.Module):
    def __init__(self, cardinalities: list[int], bucket_counts: list[int]) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.bucket_counts = bucket_counts

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [
                F.one_hot(x[:, i] % bucket_count, bucket_count)
                for i, bucket_count in enumerate(self.bucket_counts)
            ],
            dim=1,
        ).float()


def build_cat_embedding_v4(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    cap = max(8, int(trial_params['d_embedding']) * 4)
    bucket_counts = [min(cardinality, cap) for cardinality in cat_cardinalities]
    return (
        HashedOneHotEncoding(cat_cardinalities, bucket_counts),
        sum(bucket_counts),
        {'cardinalities': cat_cardinalities, 'bucket_counts': bucket_counts},
    )
