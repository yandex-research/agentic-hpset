# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v4``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _key(value: object) -> object:
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return ('__nan__',)
    return value


def categorical_preprocess_v4(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'top_k': 64, 'mappings': None})
    n_features = x_cat['train'].shape[1]
    encoded = {
        part: np.empty((values.shape[0], n_features), dtype=np.int64)
        for part, values in x_cat.items()
    }
    mappings: list[dict[object, int]] = []
    fallback_ids: list[int] = []
    for column in range(n_features):
        train_column = x_cat['train'][:, column]
        values, counts = np.unique(train_column, return_counts=True)
        order = np.argsort(-counts, kind='stable')
        kept = values[order[:64]]
        mapping = {_key(value): idx for idx, value in enumerate(kept)}
        fallback_id = len(mapping)
        mappings.append(mapping)
        fallback_ids.append(fallback_id)
        for part, part_values in x_cat.items():
            encoded[part][:, column] = [
                mapping.get(_key(value), fallback_id)
                for value in part_values[:, column]
            ]
    return (encoded, {'top_k': 64, 'mappings': mappings, 'fallback_ids': fallback_ids})
