from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class BilinearInteractionEmbedding(nn.Module):
    """Low-rank bilinear interaction embedding for numerical features.

    Projects features to a low-rank space (rank r), computes pairwise
    interactions via outer product of the projection, then linearly maps
    to d_embedding. Explicitly models second-order feature interactions
    like Factorization Machines, but embedded inside the neural pipeline.

    This captures multiplicative relationships (e.g., area = width * height)
    that additive MLPs struggle with. The low-rank constraint keeps the
    parameter count manageable: O(n_features * rank) instead of O(n_features^2).
    """

    def __init__(self, n_features: int, rank: int, d_embedding: int) -> None:
        super().__init__()
        self.rank = rank
        # Per-feature linear embedding (like v2)
        self.linear_weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.linear_bias = nn.Parameter(torch.empty(n_features, d_embedding))
        bound = d_embedding**-0.5
        nn.init.uniform_(self.linear_weight, -bound, bound)
        nn.init.uniform_(self.linear_bias, -bound, bound)
        # Low-rank projection for interactions: (n_features, rank)
        self.proj = nn.Parameter(torch.empty(n_features, rank))
        nn.init.kaiming_uniform_(self.proj, a=5**0.5)
        # Map interaction features (rank*(rank+1)/2) → d_embedding for the global token
        interaction_dim = rank * (rank + 1) // 2
        self.interaction_linear = nn.Linear(interaction_dim, d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        # Per-feature linear embedding: (batch, n_features, d_embedding)
        per_feature = torch.addcmul(self.linear_bias, self.linear_weight, x[..., None])
        # Interaction: project to rank space: (batch, rank)
        z = x @ self.proj  # (batch, rank)
        # Compute upper-triangular outer product (unique pairwise interactions)
        # z_i * z_j for i <= j
        idx_i, idx_j = torch.triu_indices(self.rank, self.rank, device=x.device)
        interactions = z[:, idx_i] * z[:, idx_j]  # (batch, rank*(rank+1)/2)
        # Map to d_embedding and add as an extra "interaction token"
        interaction_emb = self.interaction_linear(interactions)  # (batch, d_embedding)
        # Append interaction embedding as an extra feature
        # per_feature: (batch, n_features, d_embedding)
        # interaction_emb: (batch, 1, d_embedding)
        combined = torch.cat(
            [per_feature, interaction_emb.unsqueeze(1)], dim=1
        )  # (batch, n_features+1, d_embedding)
        return combined


def build_num_embedding_v5(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {}
    n_features = dataset_meta.n_num_features
    d_embedding = int(trial_params["d_embedding"])
    rank = min(max(d_embedding // 2, 2), n_features)  # rank bounded by n_features
    embedding = BilinearInteractionEmbedding(n_features, rank, d_embedding)
    # Output: (n_features + 1) * d_embedding (extra interaction token)
    output_dim = (n_features + 1) * d_embedding
    return embedding, output_dim, {"rank": rank}
