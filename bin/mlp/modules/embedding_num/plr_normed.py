"""PLR with input BatchNorm and post-encoding LayerNorm.

PLR is sensitive to the absolute scale of the input because bin boundaries
are fixed at training time. This variant adds (a) a 1-D BatchNorm on raw
numeric inputs (running stats, so val/test see the moving average) before
the encoding to align distributions, and (b) a LayerNorm across the bin
axis after encoding to keep activations well-conditioned for downstream
mixers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .piecewise_linear import LinearEmbeddings, PiecewiseLinearEncoding, compute_bins


class NormalizedPLREmbeddings(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        n_features = len(bins)
        self.input_bn = nn.BatchNorm1d(n_features)
        self.linear0 = LinearEmbeddings(n_features, d_embedding)
        self.encoding = PiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(
            torch.zeros(n_features, self.encoding.weight.shape[1], d_embedding)
        )
        self.post_ln = nn.LayerNorm(self.encoding.weight.shape[1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_bn(x)
        x_linear = self.linear0(x)
        x_ple = self.post_ln(self.encoding(x))
        x_ple = x_ple.transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return x_linear + x_ple


def build_num_embedding_v11(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bins(x_num_train, int(trial_params["n_bins"]))
    embedding = NormalizedPLREmbeddings(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {"bins": bins}
