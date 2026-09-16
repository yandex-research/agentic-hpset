"""Entity embeddings with sqrt-cardinality dimensionality.

Per-column `nn.Embedding` where each column's embedding dimension is the
heuristic `min(cap, max(2, round(sqrt(cardinality))))`. High-cardinality
columns get more capacity; low-cardinality columns stay compact. Cleaner
parameter-efficiency story than uniform `d_embedding`. Padding index =
`cardinality` carries an explicit "unknown" embedding (zero-initialized).
Distinct from `entity.py` (which uses Guo & Berkhahn's `card^0.25 * 6`
formula plus GroupNorm).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


def _embedding_dim(cardinality: int, cap: int) -> int:
    if cardinality <= 1:
        return 1
    return min(cap, max(2, int(round(cardinality ** 0.5))))


class SqrtEntityEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], dims: list[int]) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality + 1, dim, padding_idx=cardinality)
                for cardinality, dim in zip(cardinalities, dims)
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[embedding.padding_idx].zero_()

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, (cardinality, embedding) in enumerate(zip(self.cardinalities, self.embeddings)):
            values = x[:, i].clamp(0, cardinality)
            pieces.append(embedding(values))
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v7(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": [], "dims": []}
    cap = max(2, min(32, int(trial_params.get("d_embedding", 16))))
    dims = [_embedding_dim(cardinality, cap) for cardinality in cat_cardinalities]
    return (
        SqrtEntityEmbeddings(cat_cardinalities, dims),
        sum(dims),
        {"cardinalities": cat_cardinalities, "dims": dims},
    )
