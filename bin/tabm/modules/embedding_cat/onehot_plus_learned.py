from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OneHotPlusLearnedEmbedding(nn.Module):
    def __init__(self, cardinalities: list[int], d_dense: int) -> None:
        super().__init__()
        self.cardinalities = [int(cardinality) for cardinality in cardinalities]
        self.d_dense = int(d_dense)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, self.d_dense) for cardinality in self.cardinalities]
        )
        bound = self.d_dense**-0.5
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -bound, bound)
        self.d_features = [cardinality + 1 + self.d_dense for cardinality in self.cardinalities]

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for feature_idx, cardinality in enumerate(self.cardinalities):
            one_hot = F.one_hot(x[:, feature_idx], cardinality + 1).float()
            dense = self.embeddings[feature_idx](x[:, feature_idx])
            pieces.append(torch.cat([one_hot, dense], dim=1))
        return torch.cat(pieces, dim=1)


def build_cat_embedding_v4(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": []}
    d_dense = int(trial_params["d_embedding"])
    embedding = OneHotPlusLearnedEmbedding(cat_cardinalities, d_dense)
    return embedding, sum(embedding.d_features), {
        "cardinalities": cat_cardinalities,
        "d_dense": d_dense,
    }


__all__ = ["OneHotPlusLearnedEmbedding", "build_cat_embedding_v4"]
