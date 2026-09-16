# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v5``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class SmoothThresholdEmbeddings(nn.Module):
    def __init__(self, thresholds: Tensor, scales: Tensor) -> None:
        super().__init__()
        self.register_buffer('thresholds', thresholds)
        self.register_buffer('scales', scales)
        self.weight = nn.Parameter(torch.empty_like(thresholds))
        self.bias = nn.Parameter(torch.zeros_like(thresholds))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return (
            torch.sigmoid((x[..., None] - self.thresholds) / self.scales) * self.weight
            + self.bias
        )


def compute_thresholds_and_scales(x: Tensor, d_embedding: int) -> tuple[Tensor, Tensor]:
    quantiles = torch.linspace(0.0, 1.0, d_embedding, device=x.device, dtype=x.dtype)
    thresholds = torch.quantile(x, quantiles, dim=0).T.contiguous()
    q10 = torch.quantile(x, 0.1, dim=0)
    q90 = torch.quantile(x, 0.9, dim=0)
    scale = (q90 - q10).abs() / max(d_embedding - 1, 1)
    fallback = x.std(dim=0).clamp_min(torch.finfo(x.dtype).eps) / max(
        d_embedding - 1, 1
    )
    scale = torch.where(scale > 1e-06, scale, fallback).clamp_min(0.001)
    return (thresholds, scale.unsqueeze(1).expand_as(thresholds).contiguous())


def build_num_embedding_v5(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'thresholds': None, 'scales': None})
    thresholds, scales = compute_thresholds_and_scales(
        x_num_train, int(trial_params['d_embedding'])
    )
    embedding = SmoothThresholdEmbeddings(thresholds, scales)
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'thresholds': thresholds, 'scales': scales})
