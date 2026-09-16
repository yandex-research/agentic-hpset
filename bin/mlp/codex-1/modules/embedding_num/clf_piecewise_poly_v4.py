# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v4``."""

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


class PiecewiseLinearEncoding(nn.Module):
    def __init__(self, bins: list[Tensor]) -> None:
        super().__init__()
        n_features = len(bins)
        n_bins = [len(edges) - 1 for edges in bins]
        max_n_bins = max(n_bins)
        device = bins[0].device
        dtype = bins[0].dtype
        self.register_buffer(
            'weight', torch.zeros(n_features, max_n_bins, device=device, dtype=dtype)
        )
        self.register_buffer(
            'bias', torch.zeros(n_features, max_n_bins, device=device, dtype=dtype)
        )
        self.register_buffer(
            'single_bin_mask',
            torch.tensor(n_bins, dtype=torch.long, device=device) == 1,
        )
        self.register_buffer(
            'mask',
            None
            if len(set(n_bins)) == 1
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(max(n - 1, 0), dtype=torch.bool, device=device),
                            torch.zeros(
                                max_n_bins - n, dtype=torch.bool, device=device
                            ),
                            torch.ones(1, dtype=torch.bool, device=device),
                        ]
                    )
                    for n in n_bins
                ]
            ),
        )
        for i, edges in enumerate(bins):
            widths = edges.diff().clamp_min(torch.finfo(edges.dtype).eps)
            weights = 1.0 / widths
            biases = -edges[:-1] / widths
            self.weight[i, -1] = weights[-1]
            self.bias[i, -1] = biases[-1]
            if n_bins[i] > 1:
                self.weight[i, : n_bins[i] - 1] = weights[:-1]
                self.bias[i, : n_bins[i] - 1] = biases[:-1]

    def forward(self, x: Tensor) -> Tensor:
        encoded = torch.addcmul(self.bias, self.weight, x[..., None])
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
        if self.mask is not None:
            encoded = encoded * self.mask
        return encoded


class PiecewisePolyEmbeddings(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.linear0 = LinearEmbeddings(len(bins), d_embedding)
        self.encoding = PiecewiseLinearEncoding(bins)
        self.piecewise_weight = nn.Parameter(
            torch.zeros(len(bins), self.encoding.weight.shape[1], d_embedding)
        )
        self.poly_weight = nn.Parameter(torch.empty(len(bins), 3, d_embedding))
        nn.init.normal_(self.poly_weight, std=d_embedding ** (-0.5))

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.piecewise_weight).transpose(0, 1)
        poly = torch.stack([x, x.square(), torch.tanh(x)], dim=-1)
        x_poly = torch.einsum('bfk,fkd->bfd', poly, self.poly_weight)
        return x_linear + x_ple + 0.1 * x_poly


def compute_bins(x: Tensor | None, n_bins: int) -> list[Tensor]:
    if x is None:
        return []
    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device, dtype=x.dtype)
    bins: list[Tensor] = []
    for edges in torch.quantile(x, quantiles, dim=0).T:
        unique = edges.unique()
        if len(unique) < 2:
            unique = torch.stack(
                [
                    edges[0],
                    edges[0] + torch.tensor(1e-06, device=x.device, dtype=x.dtype),
                ]
            )
        bins.append(unique)
    return bins


def build_num_embedding_v4(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'bins': []})
    bins = compute_bins(x_num_train, int(trial_params['n_bins']))
    embedding = PiecewisePolyEmbeddings(bins, int(trial_params['d_embedding']))
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'bins': bins})
