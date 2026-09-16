# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v3``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class LearnablePLE(nn.Module):
    """Piecewise linear encoding with learnable bin edges.

    Uses uniform n_bins per feature. Edges are parameterized as
    edge_0 + cumulative_softplus(deltas) so they remain monotonically
    increasing during training.
    """

    def __init__(self, init_edges: list[Tensor], n_bins: int) -> None:
        super().__init__()
        n_features = len(init_edges)
        first_edges = torch.zeros(n_features)
        log_deltas = torch.zeros(n_features, n_bins)
        for i, edges in enumerate(init_edges):
            edges = edges.detach().to(dtype=torch.float32, device='cpu')
            if edges.numel() < 2:
                edges = torch.tensor([-1.0, 1.0], dtype=torch.float32)
            if edges.numel() - 1 == n_bins:
                widths = edges.diff().clamp_min(1e-06)
            else:
                lo = edges.min().item()
                hi = edges.max().item()
                if hi <= lo:
                    hi = lo + 1.0
                widths = torch.full(
                    (n_bins,), (hi - lo) / n_bins, dtype=torch.float32
                ).clamp_min(1e-06)
                edges = torch.tensor(
                    [lo + (hi - lo) * k / n_bins for k in range(n_bins + 1)],
                    dtype=torch.float32,
                )
            first_edges[i] = edges[0]
            log_deltas[i] = torch.log(torch.expm1(widths))
        self.first_edge = nn.Parameter(first_edges)
        self.log_deltas = nn.Parameter(log_deltas)
        self.n_bins = n_bins

    def edges(self) -> Tensor:
        deltas = F.softplus(self.log_deltas) + 1e-06
        cum = torch.cumsum(deltas, dim=-1)
        return torch.cat(
            [self.first_edge[..., None], self.first_edge[..., None] + cum], dim=-1
        )

    def forward(self, x: Tensor) -> Tensor:
        edges = self.edges()
        widths = edges[:, 1:] - edges[:, :-1]
        widths = widths.clamp_min(1e-06)
        weights = 1.0 / widths
        biases = -edges[:, :-1] * weights
        encoded = x[..., None] * weights + biases
        if self.n_bins == 1:
            return encoded
        last = encoded[..., -1:].clamp_min(0.0)
        encoded = torch.cat(
            [encoded[..., :1].clamp_max(1.0), encoded[..., 1:-1].clamp(0.0, 1.0), last],
            dim=-1,
        )
        return encoded


class LinearEmbeddingsV3(nn.Module):
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
    def __init__(self, init_edges: list[Tensor], n_bins: int, d_embedding: int) -> None:
        super().__init__()
        n_features = len(init_edges)
        self.linear0 = LinearEmbeddingsV3(n_features, d_embedding)
        self.encoding = LearnablePLE(init_edges, n_bins)
        self.linear = nn.Parameter(torch.zeros(n_features, n_bins, d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return x_linear + x_ple


def compute_initial_edges(x: Tensor | None, n_bins: int) -> list[Tensor]:
    if x is None:
        return []
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
    n_bins = int(trial_params['n_bins'])
    init_edges = compute_initial_edges(x_num_train, n_bins)
    embedding = LearnablePLREmbeddings(
        init_edges, n_bins, int(trial_params['d_embedding'])
    )
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'bins': init_edges})
