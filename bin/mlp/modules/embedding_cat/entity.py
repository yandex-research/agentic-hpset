"""Entity embeddings with cardinality-tied dimensionality.

Per-column `nn.Embedding` whose dimensionality scales with cardinality
(Guo & Berkhahn 2016 rule: `min(d_max, ceil(cardinality ** 0.25 * 6))`)
rather than using a fixed `d_embedding` for all columns. High-cardinality
columns get more capacity; low-cardinality columns stay compact. A
GroupNorm over the concatenated activations stabilizes training.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


def _entity_dim(cardinality: int, d_max: int) -> int:
    if cardinality <= 1:
        return 1
    return int(min(d_max, max(2, math.ceil((cardinality ** 0.25) * 6))))


class EntityCategoricalEmbedding(nn.Module):
    def __init__(self, cardinalities: list[int], d_max: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.dims: list[int] = []
        for c in cardinalities:
            d = _entity_dim(c, d_max)
            emb = nn.Embedding(c + 1, d)
            nn.init.normal_(emb.weight, std=d ** -0.5)
            self.embeddings.append(emb)
            self.dims.append(d)
        total = sum(self.dims)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=max(total, 1))
        self.total = total

    def forward(self, x: Tensor) -> Tensor:
        pieces = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        out = torch.cat(pieces, dim=1)
        return self.norm(out)


def build_cat_embedding_v6(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": []}
    d_max = int(trial_params["d_embedding"])
    embedding = EntityCategoricalEmbedding(cat_cardinalities, d_max)
    return (
        embedding,
        embedding.total,
        {"cardinalities": cat_cardinalities, "d_max": d_max, "dims": embedding.dims},
    )
