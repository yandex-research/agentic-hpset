"""Hashing-trick categorical embedding.

Each category is routed through a stable hash modulo a fixed bucket count.
Two hash functions ("double hashing") are used and their embeddings are
summed to mitigate collision noise. The parameter count stays fixed
regardless of cardinality, which is helpful for very high-cardinality
columns where one-hot blows up the input dimension.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class HashedCategoricalEmbedding(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        d_embedding: int,
        n_buckets: int = 256,
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.n_buckets = n_buckets
        n_features = len(cardinalities)
        self.table_a = nn.Embedding(n_features * n_buckets, d_embedding)
        self.table_b = nn.Embedding(n_features * n_buckets, d_embedding)
        nn.init.normal_(self.table_a.weight, std=(2 * d_embedding) ** -0.5)
        nn.init.normal_(self.table_b.weight, std=(2 * d_embedding) ** -0.5)
        self.register_buffer("primes_a", torch.tensor([2654435761], dtype=torch.long))
        self.register_buffer("primes_b", torch.tensor([40503], dtype=torch.long))
        self.d_embedding = d_embedding

    def _hash(self, x: Tensor, prime: Tensor, salt: int) -> Tensor:
        n_feat = x.shape[1]
        feat_offsets = torch.arange(n_feat, device=x.device, dtype=torch.long) * salt
        h = ((x + feat_offsets.unsqueeze(0)) * prime) % int(self.n_buckets)
        feat_table_offset = (
            torch.arange(n_feat, device=x.device, dtype=torch.long) * int(self.n_buckets)
        )
        return h + feat_table_offset.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        h_a = self._hash(x, self.primes_a, salt=982451653)
        h_b = self._hash(x, self.primes_b, salt=87178291199)
        e_a = self.table_a(h_a)
        e_b = self.table_b(h_b)
        return (e_a + e_b).flatten(1)


def build_cat_embedding_v4(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": []}
    d_embedding = int(trial_params["d_embedding"])
    n_buckets = max(64, min(1024, max(cat_cardinalities) * 2))
    output_dim = len(cat_cardinalities) * d_embedding
    return (
        HashedCategoricalEmbedding(cat_cardinalities, d_embedding, n_buckets),
        output_dim,
        {"cardinalities": cat_cardinalities, "n_buckets": n_buckets},
    )
