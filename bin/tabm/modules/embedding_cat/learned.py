from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class LearnedCategoricalEmbedding(nn.Module):
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.n_features = len(cardinalities)
        self.d_embedding = int(d_embedding)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinality) + 1, self.d_embedding) for cardinality in cardinalities]
        )
        bound = self.d_embedding**-0.5
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -bound, bound)
        self.d_features = [self.d_embedding] * self.n_features

    def get_output_shape(self) -> torch.Size:
        return torch.Size((self.n_features, self.d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack(
            [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)], dim=1
        )


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": []}
    d_embedding = int(trial_params["d_embedding"])
    embedding = LearnedCategoricalEmbedding(cat_cardinalities, d_embedding)
    return embedding, len(cat_cardinalities) * d_embedding, {
        "cardinalities": cat_cardinalities,
        "d_embedding": d_embedding,
    }


__all__ = ["LearnedCategoricalEmbedding", "build_cat_embedding_v2"]
