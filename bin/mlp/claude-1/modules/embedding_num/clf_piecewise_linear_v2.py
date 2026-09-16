# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class LinearEmbeddingsV2(nn.Module):
    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        bound = d_embedding ** (-0.5)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.addcmul(self.bias, self.weight, x[..., None])


class PiecewiseLinearEncodingV2(nn.Module):
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


class PLRTreeEmbeddings(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.linear0 = LinearEmbeddingsV2(len(bins), d_embedding)
        self.encoding = PiecewiseLinearEncodingV2(bins)
        self.linear = nn.Parameter(
            torch.zeros(len(bins), self.encoding.weight.shape[1], d_embedding)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return x_linear + x_ple


def _quantile_edges(col: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    return np.unique(np.quantile(col, qs))


def _tree_edges(col: np.ndarray, y: np.ndarray, n_bins: int, seed: int) -> np.ndarray:
    if len(np.unique(col)) <= 1:
        return _quantile_edges(col, n_bins)
    tree = DecisionTreeClassifier(max_leaf_nodes=max(n_bins, 2), random_state=seed)
    tree.fit(col.reshape(-1, 1), y)
    thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
    if thresholds.size == 0:
        return _quantile_edges(col, n_bins)
    edges = np.unique(np.concatenate([[col.min()], thresholds, [col.max()]]))
    if edges.size < 2:
        return _quantile_edges(col, n_bins)
    return edges


def compute_bins_tree(
    x: Tensor | None, y: np.ndarray, n_bins: int, seed: int
) -> list[Tensor]:
    if x is None:
        return []
    x_np = x.detach().cpu().numpy()
    bins: list[Tensor] = []
    for j in range(x_np.shape[1]):
        edges_np = _tree_edges(x_np[:, j], y, n_bins, seed)
        bins.append(torch.tensor(edges_np, device=x.device, dtype=x.dtype))
    return bins


def build_num_embedding_v2(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'bins': []})
    y_train = np.asarray(dataset_meta.y['train']).reshape(-1).astype(np.int64)
    seed = 0
    bins = compute_bins_tree(x_num_train, y_train, int(trial_params['n_bins']), seed)
    embedding = PLRTreeEmbeddings(bins, int(trial_params['d_embedding']))
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'bins': bins})
