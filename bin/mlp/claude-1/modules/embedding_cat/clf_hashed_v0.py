# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class HashedCategoricalEmbedding(nn.Module):
    """Hash-trick categorical embedding.

    Each categorical feature gets K independent embedding tables of size
    `bucket_size`. The category id is hashed K times (with K different prime
    multipliers) into one bucket per table; the K resulting embedding vectors
    are summed. This bounds memory regardless of cardinality and is robust
    to unseen categories at inference time.
    """

    def __init__(
        self,
        cardinalities: list[int],
        d_embedding: int,
        n_hash: int = 8,
        bucket_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.n_hash = n_hash
        self.bucket_size = bucket_size
        self.d_embedding = d_embedding
        self.tables = nn.ModuleList(
            [
                nn.Embedding(bucket_size, d_embedding)
                for _ in range(len(cardinalities) * n_hash)
            ]
        )
        for table in self.tables:
            nn.init.normal_(table.weight, std=(d_embedding * n_hash) ** (-0.5))
        primes = [
            2654435761,
            40503,
            1597334677,
            2246822519,
            3266489917,
            668265263,
            374761393,
            3326489917,
            2147483647,
            1000000007,
            16777619,
            2166136261,
            805306457,
            49979687,
            73856093,
            19349663,
        ]
        n_pairs = len(cardinalities) * n_hash
        mults = [primes[(seed + i) % len(primes)] for i in range(n_pairs)]
        self.register_buffer(
            'multipliers',
            torch.tensor(mults, dtype=torch.long).view(len(cardinalities), n_hash),
        )

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        n_features = len(self.cardinalities)
        for j in range(n_features):
            col = x[:, j]
            agg = None
            for h in range(self.n_hash):
                mult = self.multipliers[j, h]
                bucket = (col * mult).remainder(self.bucket_size).long()
                emb = self.tables[j * self.n_hash + h](bucket)
                agg = emb if agg is None else agg + emb
            outs.append(agg)
        return torch.cat(outs, dim=1)


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    bucket_size = max(64, min(1024, max(cat_cardinalities) * 2))
    module = HashedCategoricalEmbedding(
        cat_cardinalities, d_embedding, n_hash=8, bucket_size=bucket_size
    )
    output_dim = len(cat_cardinalities) * d_embedding
    return (
        module,
        output_dim,
        {
            'cardinalities': cat_cardinalities,
            'd_embedding': d_embedding,
            'bucket_size': bucket_size,
        },
    )
