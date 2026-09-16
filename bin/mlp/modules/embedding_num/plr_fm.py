"""PLR with a low-rank Factorization-Machine feature cross.

Adds an explicit FM-style pairwise interaction between feature embeddings
on top of PLR. Computed via the sum-of-squares trick
`0.5 * ((sum_i v_i)^2 - sum_i v_i^2)` so the cost stays O(n_features) per
batch element. A learnable per-feature gate `cross_gate` controls how much
of the shared interaction is fed back to each feature.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .piecewise_linear import LinearEmbeddings, PiecewiseLinearEncoding, compute_bins


class PLRWithFMCross(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        n_features = len(bins)
        self.linear0 = LinearEmbeddings(n_features, d_embedding)
        self.encoding = PiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(
            torch.zeros(n_features, self.encoding.weight.shape[1], d_embedding)
        )
        self.cross_gate = nn.Parameter(torch.zeros(n_features, d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        v = x_linear + x_ple
        sum_v = v.sum(dim=1, keepdim=True)
        sum_v_sq = (v * v).sum(dim=1, keepdim=True)
        interaction = 0.5 * (sum_v * sum_v - sum_v_sq)
        cross = interaction * self.cross_gate[None, ...]
        return v + cross


def build_num_embedding_v12(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bins(x_num_train, int(trial_params["n_bins"]))
    embedding = PLRWithFMCross(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {"bins": bins}
