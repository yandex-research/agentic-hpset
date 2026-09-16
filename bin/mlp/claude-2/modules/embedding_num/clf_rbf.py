# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v2``."""

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


class RBFEncoding(nn.Module):
    def __init__(self, centers: Tensor, log_scale_init: float = 0.0) -> None:
        super().__init__()
        self.centers = nn.Parameter(centers.clone())
        self.log_scale = nn.Parameter(torch.full(centers.shape, log_scale_init))

    def forward(self, x: Tensor) -> Tensor:
        diff = x[..., None] - self.centers
        scale = self.log_scale.exp()
        return torch.exp(-0.5 * (diff * scale) ** 2)


class RBFEmbeddings(nn.Module):
    def __init__(self, centers: Tensor, d_embedding: int, init_scale: float) -> None:
        super().__init__()
        self.linear0 = LinearEmbeddings(centers.shape[0], d_embedding)
        self.encoding = RBFEncoding(centers, log_scale_init=init_scale)
        self.linear = nn.Parameter(
            torch.zeros(centers.shape[0], centers.shape[1], d_embedding)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_rbf = self.encoding(x).transpose(0, 1)
        x_rbf = (x_rbf @ self.linear).transpose(0, 1)
        return x_linear + x_rbf


def compute_quantile_centers(x: Tensor, n_bins: int) -> Tensor:
    quantiles = torch.linspace(0.0, 1.0, n_bins + 2, device=x.device, dtype=x.dtype)[
        1:-1
    ]
    return torch.quantile(x, quantiles, dim=0).T


def build_num_embedding_v2(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'centers': None})
    n_bins = max(2, int(trial_params['n_bins']))
    centers = compute_quantile_centers(x_num_train, n_bins)
    feature_std = x_num_train.std(dim=0).clamp_min(0.001)
    init_scale = (1.0 / feature_std).log().unsqueeze(-1).expand(-1, centers.shape[1])
    init_scale_mean = float(init_scale.mean().item())
    embedding = RBFEmbeddings(
        centers, int(trial_params['d_embedding']), init_scale_mean
    )
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'centers': centers})
