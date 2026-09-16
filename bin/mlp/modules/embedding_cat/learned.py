from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class LearnedCatEmbedding(nn.Module):
    """Learned embedding lookup tables for categorical features.

    Each categorical feature gets its own embedding table. The embeddings are
    concatenated to produce the final representation. This is more expressive
    than one-hot encoding for high-cardinality features and produces a dense,
    lower-dimensional representation.
    """

    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, d_embedding) for cardinality in cardinalities]
        )
        for emb in self.embeddings:
            nn.init.kaiming_uniform_(emb.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [emb(x[:, i]) for i, emb in enumerate(self.embeddings)],
            dim=1,
        )


def build_cat_embedding_v1(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": []}
    d_embedding = int(trial_params["d_embedding"])
    module = LearnedCatEmbedding(cat_cardinalities, d_embedding)
    output_dim = len(cat_cardinalities) * d_embedding
    return module, output_dim, {"cardinalities": cat_cardinalities}
