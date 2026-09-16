# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np


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
            widths = edges.diff().clamp_min(1e-12)
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


class PLREmbeddings(nn.Module):
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


def compute_tree_bins(
    x: Tensor, y: np.ndarray, n_bins: int, task_type: str, seed: int
) -> list[Tensor]:
    x_np = x.detach().cpu().numpy()
    bins: list[Tensor] = []
    for j in range(x_np.shape[1]):
        col = x_np[:, j].reshape(-1, 1)
        unique_count = len(np.unique(col))
        if unique_count <= 1:
            edges_np = np.array([col.min(), col.max() + 1e-06], dtype=np.float64)
        else:
            tree_cls = (
                DecisionTreeClassifier
                if task_type in {'binclass', 'multiclass'}
                else DecisionTreeRegressor
            )
            tree = tree_cls(max_leaf_nodes=max(2, n_bins), random_state=seed)
            tree.fit(col, y)
            thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
            thresholds = np.unique(thresholds)
            edges_np = np.concatenate(
                [[col.min()], np.sort(thresholds), [col.max() + 1e-06]]
            )
            edges_np = np.unique(edges_np)
        bins.append(torch.as_tensor(edges_np, dtype=x.dtype, device=x.device))
    return bins


def build_num_embedding_v1(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'bins': []})
    y_train = dataset_meta.y['train']
    seed = int(trial_params.get('seed', 0))
    bins = compute_tree_bins(
        x_num_train, y_train, int(trial_params['n_bins']), dataset_meta.task_type, seed
    )
    embedding = PLREmbeddings(bins, int(trial_params['d_embedding']))
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'bins': bins})
