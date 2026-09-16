# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class HashingCatEmbedding(nn.Module):
    """Single shared embedding table indexed by hash(feature_id, value).

    Avoids blowing up memory on extremely high-cardinality features. Uses a
    per-feature multiplicative hash followed by modulo n_buckets.
    """

    def __init__(
        self, n_features: int, n_buckets: int, d_embedding: int, seed: int = 0
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_buckets = n_buckets
        self.d_embedding = d_embedding
        gen = torch.Generator().manual_seed(seed)
        coefficients = torch.randint(
            1, 2**31 - 1, (n_features,), generator=gen, dtype=torch.int64
        )
        coefficients = coefficients * 2 + 1
        self.register_buffer('coefficients', coefficients)
        self.embedding = nn.Embedding(n_buckets, d_embedding)
        bound = d_embedding ** (-0.5)
        nn.init.uniform_(self.embedding.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        coeffs = self.coefficients.unsqueeze(0)
        hashed = x.long() * coeffs % self.n_buckets
        out = self.embedding(hashed)
        return out.flatten(1)


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    max_card = max(cat_cardinalities)
    n_buckets = max(64, min(4 * max_card, 4096))
    embedding = HashingCatEmbedding(len(cat_cardinalities), n_buckets, d_embedding)
    output_dim = len(cat_cardinalities) * d_embedding
    return (
        embedding,
        output_dim,
        {'n_buckets': n_buckets, 'cardinalities': cat_cardinalities},
    )
