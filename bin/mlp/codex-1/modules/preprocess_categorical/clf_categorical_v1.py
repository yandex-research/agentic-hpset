# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _key(value: object) -> object:
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return ('__nan__',)
    return value


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'min_count': 2, 'mappings': None})
    n_features = x_cat['train'].shape[1]
    encoded = {
        part: np.empty((values.shape[0], n_features), dtype=np.int64)
        for part, values in x_cat.items()
    }
    mappings: list[dict[object, int]] = []
    rare_ids: list[int] = []
    for column in range(n_features):
        train_column = x_cat['train'][:, column]
        values, counts = np.unique(train_column, return_counts=True)
        frequent = values[counts >= 2]
        mapping = {_key(value): idx for idx, value in enumerate(frequent)}
        rare_id = len(mapping)
        mappings.append(mapping)
        rare_ids.append(rare_id)
        for part, part_values in x_cat.items():
            encoded[part][:, column] = [
                mapping.get(_key(value), rare_id) for value in part_values[:, column]
            ]
    return (encoded, {'min_count': 2, 'mappings': mappings, 'rare_ids': rare_ids})
