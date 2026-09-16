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


class PiecewiseLinearEncoding(nn.Module):
    def __init__(self, bins: list[Tensor]) -> None:
        super().__init__()
        n_features = len(bins)
        n_bins = [len(edges) - 1 for edges in bins]
        max_n_bins = max(n_bins)
        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))
        self.register_buffer(
            'single_bin_mask', torch.tensor(n_bins, dtype=torch.long) == 1
        )
        self.register_buffer(
            'mask',
            None
            if len(set(n_bins)) == 1
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(max(n - 1, 0), dtype=torch.bool),
                            torch.zeros(max_n_bins - n, dtype=torch.bool),
                            torch.ones(1, dtype=torch.bool),
                        ]
                    )
                    for n in n_bins
                ]
            ),
        )
        for i, edges in enumerate(bins):
            widths = edges.diff()
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


class PLREmbeddingsKMeans(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.linear0 = LinearEmbeddings(len(bins), d_embedding)
        self.encoding = PiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(
            torch.zeros(len(bins), self.encoding.weight.shape[1], d_embedding)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return x_linear + x_ple


def compute_kmeans_bins(
    x: Tensor | None, n_bins: int, n_iters: int = 25
) -> list[Tensor]:
    """Compute per-feature breakpoints using 1D KMeans (Lloyd-style).

    For each numerical feature, run a small 1D KMeans to find n_bins cluster
    centers, then derive sorted bin edges as (sorted_center[i] +
    sorted_center[i+1]) / 2 with the column min and max as outer edges.
    """
    if x is None:
        return []
    bins: list[Tensor] = []
    for column in torch.unbind(x, dim=1):
        col = column.reshape(-1)
        col_min = col.min()
        col_max = col.max()
        if col_min.item() == col_max.item():
            bins.append(
                torch.tensor(
                    [col_min.item(), col_max.item() + 1e-06],
                    device=col.device,
                    dtype=col.dtype,
                )
            )
            continue
        quantiles = torch.linspace(0.0, 1.0, n_bins, device=col.device, dtype=col.dtype)
        centers = torch.quantile(col, quantiles)
        for _ in range(n_iters):
            distances = (col[:, None] - centers[None, :]) ** 2
            assignments = distances.argmin(dim=1)
            new_centers = centers.clone()
            for k in range(n_bins):
                mask = assignments == k
                if mask.any():
                    new_centers[k] = col[mask].mean()
            if torch.allclose(new_centers, centers, atol=1e-07):
                centers = new_centers
                break
            centers = new_centers
        centers, _ = torch.sort(centers)
        midpoints = (centers[:-1] + centers[1:]) / 2.0
        edges = torch.cat([col_min.unsqueeze(0), midpoints, col_max.unsqueeze(0)])
        edges = edges.unique()
        bins.append(edges)
    return bins


def build_num_embedding_v2(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'bins': []})
    bins = compute_kmeans_bins(x_num_train, int(trial_params['n_bins']))
    embedding = PLREmbeddingsKMeans(bins, int(trial_params['d_embedding']))
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'bins': bins})
