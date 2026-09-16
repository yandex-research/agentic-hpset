# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v5``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LinearEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        bound = d_embedding ** (-0.5)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.addcmul(self.bias, self.weight, x[..., None])


class SoftBinEmbeddings(nn.Module):
    def __init__(self, centers: Tensor, widths: Tensor, d_embedding: int) -> None:
        super().__init__()
        self.register_buffer('centers', centers)
        self.register_buffer('widths', widths.clamp_min(0.0001))
        self.logit_scale = nn.Parameter(torch.zeros(centers.shape[0], 1))
        self.linear0 = LinearEmbeddings(centers.shape[0], d_embedding)
        self.weight = nn.Parameter(
            torch.empty(centers.shape[0], centers.shape[1], d_embedding)
        )
        nn.init.normal_(self.weight, std=d_embedding ** (-0.5))

    def forward(self, x: Tensor) -> Tensor:
        distances = ((x[..., None] - self.centers) / self.widths).abs()
        weights = torch.softmax(
            -distances * self.logit_scale.exp().clamp_max(25.0), dim=-1
        )
        return self.linear0(x) + torch.einsum('bfn,fnd->bfd', weights, self.weight)


def _centers_and_widths(x: Tensor, n_bins: int) -> tuple[Tensor, Tensor]:
    quantiles = torch.linspace(0.0, 1.0, n_bins, device=x.device, dtype=x.dtype)
    centers = torch.quantile(x, quantiles, dim=0).T.contiguous()
    span = (centers[:, -1:] - centers[:, :1]).abs().clamp_min(1.0)
    widths = span / max(n_bins - 1, 1)
    return (centers, widths.expand_as(centers))


def build_num_embedding_v5(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'centers': None, 'widths': None})
    centers, widths = _centers_and_widths(x_num_train, int(trial_params['n_bins']))
    embedding = SoftBinEmbeddings(centers, widths, int(trial_params['d_embedding']))
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'centers': centers, 'widths': widths})
