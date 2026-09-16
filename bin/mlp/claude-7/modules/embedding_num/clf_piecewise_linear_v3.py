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


class MultiResolutionPLREmbeddings(nn.Module):
    """Multi-resolution PLR: split d_embedding into G groups, each with its own bin count.

    Coarse bins capture global trends; fine bins capture local detail. The
    groups share the LinearEmbeddings input but each have their own
    piecewise-linear encoder and projection. Output is concatenated across
    groups along the embedding dim, then summed with the linear part.
    """

    def __init__(
        self,
        bins_per_group: list[list[Tensor]],
        d_per_group: list[int],
        n_features: int,
    ) -> None:
        super().__init__()
        assert len(bins_per_group) == len(d_per_group)
        self.linear0 = LinearEmbeddings(n_features, sum(d_per_group))
        self.encoders = nn.ModuleList(
            [PiecewiseLinearEncoding(bins) for bins in bins_per_group]
        )
        self.linears = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(n_features, encoder.weight.shape[1], d))
                for encoder, d in zip(self.encoders, d_per_group)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        outputs: list[Tensor] = []
        for encoder, linear in zip(self.encoders, self.linears):
            encoded = encoder(x).transpose(0, 1)
            encoded = (encoded @ linear).transpose(0, 1)
            outputs.append(encoded)
        x_ple = torch.cat(outputs, dim=-1)
        return x_linear + x_ple


def compute_bins(x: Tensor | None, n_bins: int) -> list[Tensor]:
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
        return (None, 0, {'bins_per_group': []})
    base_n_bins = max(int(trial_params['n_bins']), 4)
    base_d = int(trial_params['d_embedding'])
    n_groups = 4 if base_d >= 4 else max(base_d, 1)
    d_per_group = [base_d // n_groups] * n_groups
    for i in range(base_d - sum(d_per_group)):
        d_per_group[i] += 1
    n_bin_levels = [
        max(base_n_bins // 4, 2),
        max(base_n_bins // 2, 2),
        base_n_bins,
        max(base_n_bins * 2, 2),
    ][:n_groups]
    bins_per_group = [compute_bins(x_num_train, n) for n in n_bin_levels]
    embedding = MultiResolutionPLREmbeddings(
        bins_per_group, d_per_group, dataset_meta.n_num_features
    )
    output_dim = dataset_meta.n_num_features * sum(d_per_group)
    return (
        embedding,
        output_dim,
        {
            'bins_per_group': bins_per_group,
            'd_per_group': d_per_group,
            'n_bin_levels': n_bin_levels,
        },
    )
