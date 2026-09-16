# ruff: noqa
"""Standalone implementation for ``build_num_embedding_v4``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class AttentionBinEmbedding(nn.Module):
    """Per-feature attention over learnable bin tokens.

    For each feature j with bin centers c_{j,k}, compute attention weights
    softmax(-(x_j - c_{j,k})^2 / tau) over k, then return
    sum_k attn_{j,k} * V_{j,k} + W_j * x_j + b_j.
    """

    def __init__(
        self, bin_centers: Tensor, d_embedding: int, init_log_tau: float = 0.0
    ) -> None:
        super().__init__()
        n_features, n_bins = bin_centers.shape
        self.register_buffer('bin_centers', bin_centers)
        self.values = nn.Parameter(torch.zeros(n_features, n_bins, d_embedding))
        nn.init.normal_(self.values, std=d_embedding ** (-0.5))
        self.log_tau = nn.Parameter(torch.full((n_features,), float(init_log_tau)))
        self.linear_w = nn.Parameter(torch.empty(n_features, d_embedding))
        self.linear_b = nn.Parameter(torch.empty(n_features, d_embedding))
        bound = d_embedding ** (-0.5)
        nn.init.uniform_(self.linear_w, -bound, bound)
        nn.init.uniform_(self.linear_b, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        diff = x[..., None] - self.bin_centers
        tau = F.softplus(self.log_tau) + 0.001
        logits = -(diff**2) / tau[None, :, None]
        attn = F.softmax(logits, dim=-1)
        out = torch.einsum('bfk,fkd->bfd', attn, self.values)
        out = out + torch.addcmul(self.linear_b, self.linear_w, x[..., None])
        return out


def compute_bin_centers(x: Tensor | None, n_bins: int) -> Tensor:
    """Return tensor of shape (n_features, n_bins) of bin center values."""
    if x is None:
        return torch.empty(0, 0)
    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device, dtype=x.dtype)
    edges = torch.quantile(x, quantiles, dim=0).T
    centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
    return centers


def build_num_embedding_v4(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return (None, 0, {'bins': []})
    n_bins = int(trial_params['n_bins'])
    centers = compute_bin_centers(x_num_train, n_bins)
    embedding = AttentionBinEmbedding(centers, int(trial_params['d_embedding']))
    output_dim = dataset_meta.n_num_features * int(trial_params['d_embedding'])
    return (embedding, output_dim, {'bin_centers': centers})
