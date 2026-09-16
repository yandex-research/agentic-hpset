"""Rank-based PLR (percentile-rank inputs over fixed [0,1] bins).

Each numeric input is mapped to its percentile rank within the *training*
distribution (via `torch.searchsorted` against sorted train values per
feature), then fed to PLR with bins equally spaced on [0, 1]. The rank
mapping is robust to outliers and heavy tails. The training-time ECDF is
stored as a buffer so val/test queries are O(log n_train) per element.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .piecewise_linear import LinearEmbeddings, PiecewiseLinearEncoding


class RankEmbedder(nn.Module):
    def __init__(
        self,
        x_train: Tensor,
        n_bins: int,
        d_embedding: int,
    ) -> None:
        super().__init__()
        sorted_train, _ = torch.sort(x_train, dim=0)
        self.register_buffer("sorted_train", sorted_train)
        self.n_train = sorted_train.shape[0]
        n_features = sorted_train.shape[1]
        edge_arr = torch.linspace(0.0, 1.0, n_bins + 1, dtype=sorted_train.dtype)
        bins = [edge_arr.clone() for _ in range(n_features)]
        self.linear0 = LinearEmbeddings(n_features, d_embedding)
        self.encoding = PiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(
            torch.zeros(n_features, self.encoding.weight.shape[1], d_embedding)
        )

    def _to_rank(self, x: Tensor) -> Tensor:
        sorted_t = self.sorted_train.transpose(0, 1).contiguous()
        x_t = x.transpose(0, 1).contiguous()
        idx = torch.searchsorted(sorted_t, x_t)
        ranks = idx.float() / float(max(self.n_train, 1))
        return ranks.clamp(0.0, 1.0).transpose(0, 1)

    def forward(self, x: Tensor) -> Tensor:
        ranks = self._to_rank(x)
        x_linear = self.linear0(ranks)
        x_ple = self.encoding(ranks).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return x_linear + x_ple


def build_num_embedding_v13(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    embedding = RankEmbedder(
        x_num_train, int(trial_params["n_bins"]), int(trial_params["d_embedding"])
    )
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {"bins": []}
