from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn


def task_output_dim(dataset_meta: Any) -> int:
    if dataset_meta.is_multiclass:
        return int(dataset_meta.n_classes)
    return 1


def resolve_cat_embedding(
    cat_embedding: nn.Module | None,
    cat_cardinalities: list[int],
    default_factory: Callable[[list[int]], nn.Module],
) -> tuple[nn.Module | None, list[int]]:
    """Resolve categorical embedding and per-feature dimensions.

    ``cat_cardinalities`` are raw train-fitted cardinalities. The internal
    one-hot fallback allocates one extra unknown bucket, while external
    embeddings report their own dimensions or fall back to raw one-hot width.
    """
    cardinalities = [int(cardinality) for cardinality in cat_cardinalities]
    if not cardinalities:
        return None, []
    if cat_embedding is None:
        dims = [cardinality + 1 for cardinality in cardinalities]
        return default_factory(dims), dims
    if hasattr(cat_embedding, "d_features"):
        return cat_embedding, [int(dim) for dim in cat_embedding.d_features]
    if hasattr(cat_embedding, "get_output_shape"):
        n_features, d_feature = cat_embedding.get_output_shape()
        if int(n_features) != len(cardinalities):
            raise ValueError(
                "Cat embedding feature count does not match dataset: "
                f"{int(n_features)} != {len(cardinalities)}."
            )
        return cat_embedding, [int(d_feature)] * int(n_features)
    return cat_embedding, cardinalities
