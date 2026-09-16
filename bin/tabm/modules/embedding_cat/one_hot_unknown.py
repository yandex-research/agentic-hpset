from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OneHotUnknownEncoding(nn.Module):
    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self.cardinalities = [int(cardinality) for cardinality in cardinalities]
        self.d_features = [cardinality + 1 for cardinality in self.cardinalities]

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [
                F.one_hot(x[:, i], cardinality + 1)
                for i, cardinality in enumerate(self.cardinalities)
            ],
            dim=1,
        ).float()


def build_cat_embedding_v1(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": []}
    embedding = OneHotUnknownEncoding(cat_cardinalities)
    return embedding, sum(embedding.d_features), {"cardinalities": cat_cardinalities}


__all__ = ["OneHotUnknownEncoding", "build_cat_embedding_v1"]
