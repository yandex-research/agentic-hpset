"""PLR followed by a per-feature 2-layer MLP head.

After the standard PLR encoding, each numeric feature passes through a tiny
2-layer MLP (with ReLU) implemented as batched einsum that is unique per
feature. Gives every feature its own non-linear mapping into d_embedding-dim
space while staying parameter-efficient.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .piecewise_linear import LinearEmbeddings, PiecewiseLinearEncoding, compute_bins


class PLRWithPerFeatureMLP(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        n_features = len(bins)
        self.linear0 = LinearEmbeddings(n_features, d_embedding)
        self.encoding = PiecewiseLinearEncoding(bins)
        max_n_bins = self.encoding.weight.shape[1]
        self.W1 = nn.Parameter(torch.zeros(n_features, max_n_bins, d_embedding))
        nn.init.normal_(self.W1, std=(max_n_bins ** -0.5))
        self.b1 = nn.Parameter(torch.zeros(n_features, d_embedding))
        self.W2 = nn.Parameter(torch.zeros(n_features, d_embedding, d_embedding))
        nn.init.normal_(self.W2, std=(d_embedding ** -0.5))
        self.b2 = nn.Parameter(torch.zeros(n_features, d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x)
        h = torch.einsum("bfk,fkd->bfd", x_ple, self.W1) + self.b1
        h = F.relu(h)
        h = torch.einsum("bfk,fkd->bfd", h, self.W2) + self.b2
        return x_linear + h


def build_num_embedding_v8(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bins(x_num_train, int(trial_params["n_bins"]))
    embedding = PLRWithPerFeatureMLP(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {"bins": bins}
