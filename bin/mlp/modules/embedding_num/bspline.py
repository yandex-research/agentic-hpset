from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


def _compute_knots(x: Tensor, n_bases: int) -> Tensor:
    """Compute uniformly-spaced knots covering the data range with padding."""
    # n_bases cubic B-splines need n_bases + 4 knots (degree 3 + 1 extra each side)
    n_knots = n_bases + 4
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    span = x_max - x_min
    span = torch.where(span == 0, torch.ones_like(span), span)
    # Pad range by one knot spacing on each side
    spacing = span / (n_knots - 7)  # interior spacing
    lo = x_min - 3 * spacing
    hi = x_max + 3 * spacing
    # (n_features, n_knots)
    knots = lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * torch.linspace(
        0, 1, n_knots, device=x.device, dtype=x.dtype
    ).unsqueeze(0)
    return knots


class CubicBSplineBasis(nn.Module):
    """Evaluates cubic (degree-3) B-spline basis functions.

    Given knots t_0 < t_1 < ... < t_{n+3}, there are n basis functions B_i(x)
    of degree 3. Each basis function has local support [t_i, t_{i+4}], making
    the representation sparse and stable. Smoother than piecewise-linear and
    more expressive than polynomial features.
    """

    def __init__(self, knots: Tensor) -> None:
        super().__init__()
        # knots: (n_features, n_knots)
        self.register_buffer("knots", knots)
        self.n_bases = knots.shape[1] - 4  # degree 3

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_features)
        # knots: (n_features, n_knots)
        t = self.knots  # (F, K)
        # Expand x to (batch, F, 1) for broadcasting with knots
        x_exp = x.unsqueeze(-1)  # (B, F, 1)
        K = t.shape[1]
        # Cox-de Boor recursion, degree 0
        # B_i^0(x) = 1 if t_i <= x < t_{i+1}, else 0
        # Use (K-1) intervals
        b = ((x_exp >= t[:, :-1]) & (x_exp < t[:, 1:])).float()  # (B, F, K-1)
        # Recursion for degrees 1, 2, 3
        for d in range(1, 4):
            n_basis = K - d - 1
            left_num = x_exp - t[:, :n_basis]      # (B, F, n_basis)
            left_den = t[:, d : d + n_basis] - t[:, :n_basis]
            left_den = left_den.clamp_min(1e-8)
            left = left_num / left_den * b[..., :n_basis]
            right_num = t[:, d + 1 : d + 1 + n_basis] - x_exp
            right_den = t[:, d + 1 : d + 1 + n_basis] - t[:, 1 : 1 + n_basis]
            right_den = right_den.clamp_min(1e-8)
            right = right_num / right_den * b[..., 1 : 1 + n_basis]
            b = left + right  # (B, F, n_basis)
        return b  # (B, F, n_bases)


class BSplineEmbedding(nn.Module):
    """B-spline basis encoding + learned linear projection per feature."""

    def __init__(self, knots: Tensor, d_embedding: int) -> None:
        super().__init__()
        self.basis = CubicBSplineBasis(knots)
        n_features = knots.shape[0]
        self.linear = nn.Parameter(
            torch.empty(n_features, self.basis.n_bases, d_embedding)
        )
        nn.init.kaiming_uniform_(self.linear, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        # basis_vals: (B, F, n_bases)
        basis_vals = self.basis(x)
        # Project each feature's basis values to d_embedding
        # (B, F, n_bases) @ (F, n_bases, d_emb) -> (B, F, d_emb)
        out = torch.einsum("bfn,fnd->bfd", basis_vals, self.linear)
        return out  # (B, F, d_emb)


def build_num_embedding_v3(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {}
    n_features = dataset_meta.n_num_features
    d_embedding = int(trial_params["d_embedding"])
    n_bins = int(trial_params["n_bins"])
    # Use n_bins as the number of B-spline basis functions
    knots = _compute_knots(x_num_train, n_bins)
    embedding = BSplineEmbedding(knots, d_embedding)
    output_dim = n_features * d_embedding
    return embedding, output_dim, {"n_bases": n_bins}
