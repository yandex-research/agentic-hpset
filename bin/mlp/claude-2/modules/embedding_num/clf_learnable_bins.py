# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnablePiecewiseLinearEncoding(nn.Module):
    """Differentiable PLR: bin edges parameterized as cumulative softplus deltas
    starting from a learnable left edge. The PLR position is computed from the
    current edges so gradients flow into the binning."""

    def __init__(self, edges: Tensor) -> None:
        super().__init__()
        self.n_features, self.n_edges = edges.shape
        self.n_bins = self.n_edges - 1
        left = edges[:, 0]
        deltas = (edges[:, 1:] - edges[:, :-1]).clamp_min(1e-06)
        log_deltas = torch.log(torch.expm1(deltas).clamp_min(1e-12))
        self.left = nn.Parameter(left.clone())
        self.log_deltas = nn.Parameter(log_deltas.clone())

    def edges(self) -> Tensor:
        deltas = torch.nn.functional.softplus(self.log_deltas)
        cum = torch.cumsum(deltas, dim=-1)
        zero = torch.zeros_like(cum[..., :1])
        return self.left[..., None] + torch.cat([zero, cum], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        edges = self.edges()
        widths = (edges[..., 1:] - edges[..., :-1]).clamp_min(1e-06)
        x_b = x.transpose(0, 1)[..., None]
        left_edges = edges[..., :-1].unsqueeze(1)
        widths_b = widths.unsqueeze(1)
        pos = (x_b - left_edges) / widths_b
        first = pos[..., :1].clamp_max(1.0)
        if self.n_bins == 1:
            return first.squeeze(-2).transpose(0, 1).unsqueeze(-1)
        mid = pos[..., 1:-1].clamp(0.0, 1.0) if self.n_bins > 2 else pos[..., 1:1]
        last = pos[..., -1:].clamp_min(0.0)
        if self.n_bins == 2:
            encoded = torch.cat([first, last], dim=-1)
        else:
            encoded = torch.cat([first, mid, last], dim=-1)
        return encoded.transpose(0, 1)


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


class LearnablePLREmbeddings(nn.Module):
    def __init__(self, edges: Tensor, d_embedding: int) -> None:
        super().__init__()
        n_features, n_edges = edges.shape
        n_bins = n_edges - 1
        self.linear0 = LinearEmbeddings(n_features, d_embedding)
        self.encoding = LearnablePiecewiseLinearEncoding(edges)
        self.linear = nn.Parameter(torch.zeros(n_features, n_bins, d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return x_linear + x_ple


def compute_quantile_edges(x: Tensor, n_bins: int) -> Tensor:
    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device, dtype=x.dtype)
    return torch.quantile(x, quantiles, dim=0).T


def build_num_embedding_v4(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'edges': None})
    n_bins = max(2, int(trial_params['n_bins']))
    edges = compute_quantile_edges(x_num_train, n_bins)
    embedding = LearnablePLREmbeddings(edges, int(trial_params['d_embedding']))
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'edges': edges})
