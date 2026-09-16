from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class PolynomialEmbedding(nn.Module):
    """Per-feature polynomial basis [1, x, x^2, x^3] linearly projected to d.

    Each numeric feature is expanded into a degree-3 polynomial basis (with the
    constant term replaced by a learnable bias). A per-feature linear projection
    of shape (4, d_embedding) maps the basis to the target embedding size.
    Power-3 captures non-monotonic curvature without requiring data-dependent
    binning.
    """

    DEGREE = 3

    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.d_embedding = int(d_embedding)
        # Per-feature linear projection from (degree+1) basis -> d_embedding.
        weight = torch.empty(self.n_features, self.DEGREE + 1, self.d_embedding)
        bound = ((self.DEGREE + 1) ** -0.5)
        nn.init.uniform_(weight, -bound, bound)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(self.n_features, self.d_embedding))
        # Per-feature scale for x to bound the cubic from blowing up.
        self.input_scale = nn.Parameter(torch.ones(self.n_features))

    def get_output_shape(self) -> torch.Size:
        return torch.Size((self.n_features, self.d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (..., n_features). Apply learnable per-feature scale.
        scaled = x * self.input_scale
        # Build polynomial basis along the last new axis.
        basis = torch.stack(
            [
                torch.ones_like(scaled),
                scaled,
                scaled * scaled,
                scaled * scaled * scaled,
            ],
            dim=-1,
        )
        # basis shape: (..., n_features, degree+1).  Project each feature.
        # Use einsum-like computation: out[..., f, d] = sum_b basis[..., f, b] * weight[f, b, d]
        out = torch.einsum("...fb,fbd->...fd", basis, self.weight)
        out = out + self.bias
        return out


def build_num_embedding_v2(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    """Polynomial-basis numerical embedding (no bins).

    Hypothesis: PLR's strength is letting the model learn arbitrary 1D shapes per
    feature, but it commits to a discretization fixed at training start. A small
    polynomial basis [1, x, x^2, x^3] gives the model a fully-differentiable
    smooth function class with parameters that adapt to the data, without the
    hyperparameter coupling between n_bins and the embedding's expressivity.
    On already-standardized inputs the cubic stays bounded and well-conditioned.
    """
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {}
    embedding = PolynomialEmbedding(
        int(dataset_meta.n_num_features), int(trial_params["d_embedding"])
    )
    output_dim = int(dataset_meta.n_num_features) * int(trial_params["d_embedding"])
    return embedding, output_dim, {"d_embedding": int(trial_params["d_embedding"])}
