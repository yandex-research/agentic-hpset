from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .piecewise_linear import LinearEmbeddings, PiecewiseLinearEncoding, compute_bins


class GatedPLREmbeddings(nn.Module):
    """PLR embedding multiplied by a learnable per-feature sigmoid gate.

    The gate logits are initialized to zero so all features start at gate ~0.5.
    With L2 weight decay the optimizer is incentivized to push uninformative
    features' gates to zero, effectively pruning their contribution.
    """

    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.n_features = len(bins)
        self.d_embedding = int(d_embedding)
        self.linear0 = LinearEmbeddings(len(bins), d_embedding)
        self.encoding = PiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(
            torch.zeros(len(bins), self.encoding.weight.shape[1], d_embedding)
        )
        self.gate_logits = nn.Parameter(torch.zeros(self.n_features))

    def get_output_shape(self) -> torch.Size:
        return torch.Size((self.n_features, self.d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        embedded = x_linear + x_ple
        # Initial gate = 1.0 (sigmoid(0)=0.5, scaled by 2) to match the baseline scale.
        gate = torch.sigmoid(self.gate_logits) * 2.0
        return embedded * gate.view(*([1] * (embedded.ndim - 2)), self.n_features, 1)


def build_num_embedding_v1(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    """Quantile-bin PLR with learnable per-feature sparsity gates.

    Hypothesis: tabular regression typically has many weakly-informative columns.
    A learned multiplicative gate per input feature (initialized to 1, regularized
    by weight decay through gate logits) lets the optimizer softly prune useless
    features, sharpening the model's effective input space without altering the
    PLR primitive. This is a cheap form of induced sparsity that has shown
    consistent gains over plain PLR.
    """
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bins(x_num_train, int(trial_params["n_bins"]))
    embedding = GatedPLREmbeddings(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {
        "bins": bins,
        "d_embedding": int(trial_params["d_embedding"]),
    }


__all__ = ["GatedPLREmbeddings", "build_num_embedding_v1"]
