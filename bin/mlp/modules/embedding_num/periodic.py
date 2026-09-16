from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class PeriodicEmbedding(nn.Module):
    """Periodic (Fourier-style) embedding for numerical features.

    Each numerical feature is mapped through learnable sinusoidal functions:
        output = Linear(cat(sin(2*pi*freq*x + phase), cos(2*pi*freq*x + phase)))

    The frequencies and phases are learned per feature. This captures periodic
    patterns and provides a rich, high-frequency representation similar to
    positional encodings in transformers.

    Based on "On Embeddings for Numerical Features in Tabular Deep Learning"
    (Gorishniy et al., NeurIPS 2022).
    """

    def __init__(self, n_features: int, n_frequencies: int, d_embedding: int) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.empty(n_features, n_frequencies))
        self.phases = nn.Parameter(torch.empty(n_features, n_frequencies))
        self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        self._init_weights(n_frequencies)

    def _init_weights(self, n_frequencies: int) -> None:
        nn.init.uniform_(self.frequencies, 0.0, math.log(100.0) / (2.0 * math.pi))
        nn.init.uniform_(self.phases, 0.0, 2.0 * math.pi)
        nn.init.kaiming_uniform_(self.linear.weight, a=5**0.5)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_features)
        # -> (batch, n_features, n_frequencies)
        angles = 2.0 * math.pi * self.frequencies * x[..., None] + self.phases
        # -> (batch, n_features, 2*n_frequencies)
        periodic = torch.cat([angles.sin(), angles.cos()], dim=-1)
        # -> (batch, n_features, d_embedding)
        return self.linear(periodic)


def build_num_embedding_v1(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {}
    n_features = dataset_meta.n_num_features
    d_embedding = int(trial_params["d_embedding"])
    n_frequencies = max(d_embedding // 2, 4)
    embedding = PeriodicEmbedding(n_features, n_frequencies, d_embedding)
    output_dim = n_features * d_embedding
    return embedding, output_dim, {"n_frequencies": n_frequencies}
