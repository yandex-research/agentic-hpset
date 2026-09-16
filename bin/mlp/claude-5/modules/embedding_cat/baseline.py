# ruff: noqa
# Source: checked-in agent-ablation source
"""Standalone implementation for ``build_cat_embedding_v0``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class OneHotEncoding(nn.Module):
    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self.cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [
                F.one_hot(x[:, i], cardinality)
                for i, cardinality in enumerate(self.cardinalities)
            ],
            dim=1,
        ).float()


def build_cat_embedding_v0(
    cardinalities: list[int],
    _dataset: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cardinalities:
        return (None, 0, {'cardinalities': []})
    return (
        OneHotEncoding(cardinalities),
        sum(cardinalities),
        {'cardinalities': cardinalities},
    )
