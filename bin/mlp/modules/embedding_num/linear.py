from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class LinearNumEmbedding(nn.Module):
    """Simple per-feature linear projection for numerical features.

    Each feature gets its own weight vector and bias vector of size d_embedding.
    No binning or quantization — just a learned linear transform per feature.
    This is lightweight and can outperform PLR on datasets with smooth
    distributions where binning introduces unnecessary quantization noise.
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        bound = d_embedding**-0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_features) -> (batch, n_features, d_embedding)
        return torch.addcmul(self.bias, self.weight, x[..., None])


def build_num_embedding_v2(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {}
    n_features = dataset_meta.n_num_features
    d_embedding = int(trial_params["d_embedding"])
    embedding = LinearNumEmbedding(n_features, d_embedding)
    output_dim = n_features * d_embedding
    return embedding, output_dim, {}
