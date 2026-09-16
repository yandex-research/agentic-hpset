# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LearnablePiecewiseLinearEncoding(nn.Module):
    def __init__(self, bins: list[Tensor]) -> None:
        super().__init__()
        n_features = len(bins)
        n_bins = [len(edges) - 1 for edges in bins]
        max_n_bins = max(n_bins)
        self.n_features = n_features
        self.max_n_bins = max_n_bins
        edges_tensor = torch.zeros(n_features, max_n_bins + 1)
        gap_log_tensor = torch.full((n_features, max_n_bins), float('-inf'))
        valid_mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)
        for i, edges in enumerate(bins):
            edges_tensor[i, : edges.numel()] = edges
            edges_tensor[i, edges.numel() :] = edges[-1]
            gaps = edges.diff().clamp_min(1e-06).log()
            gap_log_tensor[i, : gaps.numel()] = gaps
            valid_mask[i, : gaps.numel()] = True
        self.register_buffer('edge_left', edges_tensor[:, 0:1].clone())
        self.gap_log = nn.Parameter(gap_log_tensor.clone())
        self.register_buffer('valid_mask', valid_mask)
        self.register_buffer(
            'single_bin_mask', torch.tensor([n == 1 for n in n_bins], dtype=torch.bool)
        )

    def _edges(self) -> Tensor:
        gap_log = self.gap_log.masked_fill(~self.valid_mask, float('-inf'))
        gaps = gap_log.exp()
        gaps = torch.where(self.valid_mask, gaps, torch.zeros_like(gaps))
        cum = gaps.cumsum(dim=1)
        edges = torch.cat([self.edge_left, self.edge_left + cum], dim=1)
        return edges

    def forward(self, x: Tensor) -> Tensor:
        edges = self._edges()
        widths = (edges[:, 1:] - edges[:, :-1]).clamp_min(1e-06)
        weights = 1.0 / widths
        biases = -edges[:, :-1] / widths
        encoded = torch.addcmul(biases, weights, x[..., None])
        if encoded.shape[-1] == 1:
            return encoded
        last = torch.where(
            self.single_bin_mask[..., None],
            encoded[..., -1:],
            encoded[..., -1:].clamp_min(0.0),
        )
        encoded = torch.cat(
            [encoded[..., :1].clamp_max(1.0), encoded[..., 1:-1].clamp(0.0, 1.0), last],
            dim=-1,
        )
        encoded = encoded * self.valid_mask.float() + 0.0
        valid_with_last = self.valid_mask.clone()
        valid_with_last[:, -1] = True
        encoded = encoded * valid_with_last.float()
        return encoded


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


class PLREmbeddings(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.linear0 = LinearEmbeddings(len(bins), d_embedding)
        self.encoding = LearnablePiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(
            torch.zeros(len(bins), self.encoding.max_n_bins, d_embedding)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return x_linear + x_ple


def compute_quantile_bins(x: Tensor, n_bins: int) -> list[Tensor]:
    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device, dtype=x.dtype)
    return [q.unique() for q in torch.quantile(x, quantiles, dim=0).T]


def build_num_embedding_v2(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'bins': []})
    bins = compute_quantile_bins(x_num_train, int(trial_params['n_bins']))
    embedding = PLREmbeddings(bins, int(trial_params['d_embedding']))
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'bins': bins})
