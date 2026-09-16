# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class SmoothThresholdEmbeddings(nn.Module):
    def __init__(self, thresholds: Tensor, scales: Tensor) -> None:
        super().__init__()
        self.register_buffer('thresholds', thresholds)
        self.register_buffer('scales', scales.clamp_min(0.0001))
        self.weight = nn.Parameter(torch.empty_like(thresholds))
        self.bias = nn.Parameter(torch.empty_like(thresholds))
        bound = thresholds.shape[1] ** (-0.5)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        basis = torch.sigmoid((x[..., None] - self.thresholds) / self.scales)
        return basis * self.weight + self.bias


def _thresholds_and_scales(x: Tensor, d_embedding: int) -> tuple[Tensor, Tensor]:
    quantiles = torch.linspace(0.02, 0.98, d_embedding, device=x.device, dtype=x.dtype)
    thresholds = torch.quantile(x, quantiles, dim=0).T.contiguous()
    diffs = thresholds.diff(dim=1).abs()
    fallback = x.std(dim=0, unbiased=False).clamp_min(0.001)
    median_diff = torch.median(diffs, dim=1).values if diffs.shape[1] else fallback
    scale = torch.where(median_diff > 0.0001, median_diff, fallback)
    return (thresholds, scale[:, None].expand_as(thresholds).contiguous())


def build_num_embedding_v4(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'thresholds': None, 'scales': None})
    d_embedding = int(trial_params['d_embedding'])
    thresholds, scales = _thresholds_and_scales(x_num_train, d_embedding)
    return (
        SmoothThresholdEmbeddings(thresholds, scales),
        dataset_meta.n_num_features * d_embedding,
        {'thresholds': thresholds, 'scales': scales},
    )
