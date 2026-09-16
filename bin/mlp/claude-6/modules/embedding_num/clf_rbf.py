# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v3``."""

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
    def __init__(self, bins: list[Tensor], n_centers: int) -> None:
        super().__init__()
        n_features = len(bins)
        centers = torch.zeros(n_features, n_centers)
        log_widths = torch.zeros(n_features, n_centers)
        for i, edges in enumerate(bins):
            mids = 0.5 * (edges[:-1] + edges[1:])
            if mids.numel() == 0:
                mids = edges[:1]
            if mids.numel() < n_centers:
                lo, hi = (edges[0].item(), edges[-1].item())
                if hi <= lo:
                    hi = lo + 1.0
                mids = torch.linspace(lo, hi, n_centers, dtype=edges.dtype)
            else:
                idx = torch.linspace(0, mids.numel() - 1, n_centers).round().long()
                mids = mids[idx]
            centers[i] = mids
            spacing = (edges[-1] - edges[0]).clamp_min(1e-06) / max(n_centers, 1)
            log_widths[i] = torch.log(spacing.expand(n_centers).clone())
        self.centers = nn.Parameter(centers)
        self.log_widths = nn.Parameter(log_widths)

    def forward(self, x: Tensor) -> Tensor:
        diff = x[..., None] - self.centers
        widths = self.log_widths.exp().clamp_min(1e-06)
        return torch.exp(-0.5 * (diff / widths) ** 2)


class RBFLinearEmbeddings(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int, n_centers: int) -> None:
        super().__init__()
        self.linear0 = LinearEmbeddings(len(bins), d_embedding)
        self.encoding = RBFEncoding(bins, n_centers)
        self.linear = nn.Parameter(torch.zeros(len(bins), n_centers, d_embedding))
        nn.init.normal_(self.linear, std=n_centers ** (-0.5))

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_rbf = self.encoding(x).transpose(0, 1)
        x_rbf = (x_rbf @ self.linear).transpose(0, 1)
        return x_linear + x_rbf


def compute_quantile_bins(x: Tensor, n_bins: int) -> list[Tensor]:
    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device, dtype=x.dtype)
    return [q.unique() for q in torch.quantile(x, quantiles, dim=0).T]


def build_num_embedding_v3(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'bins': []})
    bins = compute_quantile_bins(x_num_train, int(trial_params['n_bins']))
    n_centers = max(int(trial_params['n_bins']), 2)
    embedding = RBFLinearEmbeddings(bins, int(trial_params['d_embedding']), n_centers)
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'bins': bins})
