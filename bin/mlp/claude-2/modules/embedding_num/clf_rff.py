# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v5``."""

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


class RandomFourierFeatures(nn.Module):
    def __init__(self, n_features: int, n_freq: int, sigma: float, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device='cpu').manual_seed(seed)
        freqs = torch.randn(n_features, n_freq, generator=gen) * sigma
        phase = torch.rand(n_features, n_freq, generator=gen) * 2.0 * 3.141592653589793
        self.register_buffer('freqs', freqs)
        self.register_buffer('phase', phase)

    def forward(self, x: Tensor) -> Tensor:
        proj = x.unsqueeze(-1) * self.freqs.unsqueeze(0) + self.phase.unsqueeze(0)
        return torch.cos(proj)


class RFFEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_freq: int, d_embedding: int, sigma: float, seed: int
    ) -> None:
        super().__init__()
        self.linear0 = LinearEmbeddings(n_features, d_embedding)
        self.encoding = RandomFourierFeatures(n_features, n_freq, sigma, seed)
        self.linear = nn.Parameter(torch.zeros(n_features, n_freq, d_embedding))
        nn.init.kaiming_uniform_(self.linear, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_rff = self.encoding(x).transpose(0, 1)
        x_rff = (x_rff @ self.linear).transpose(0, 1)
        return x_linear + x_rff


def build_num_embedding_v5(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {})
    n_freq = max(8, int(trial_params['n_bins']))
    d_embedding = int(trial_params['d_embedding'])
    sigma = 1.0
    seed = int(trial_params.get('seed', 0)) if 'seed' in trial_params else 0
    embedding = RFFEmbeddings(
        dataset_meta.n_num_features, n_freq, d_embedding, sigma=sigma, seed=seed
    )
    output_dim = dataset_meta.n_num_features * d_embedding
    return (embedding, output_dim, {'n_freq': n_freq})
