# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _ordinal_encode(x_cat: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    unknown_value = np.iinfo(np.int64).max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=unknown_value, dtype=np.int64
    ).fit(x_cat['train'])
    encoded = {
        part: encoder.transform(values).astype(np.int64)
        for part, values in x_cat.items()
    }
    train_max = (
        encoded['train'].max(axis=0)
        if encoded['train'].shape[1]
        else np.array([], dtype=np.int64)
    )
    for part in ('val', 'test'):
        for column in range(encoded[part].shape[1]):
            mask = encoded[part][:, column] == unknown_value
            if mask.any():
                encoded[part][mask, column] = train_max[column] + 1
    return (encoded, unknown_value)


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None or x_cat['train'].shape[1] == 0:
        return (None, {'mappings': None, 'unknown_value': None})
    encoded, unknown_value = _ordinal_encode(x_cat)
    mappings: list[np.ndarray] = []
    transformed = {
        part: np.zeros_like(values, dtype=np.int64) for part, values in encoded.items()
    }
    for column in range(encoded['train'].shape[1]):
        train_col = encoded['train'][:, column]
        n_train_categories = int(train_col.max()) + 1
        counts = np.bincount(train_col, minlength=n_train_categories)
        order = np.argsort(-counts, kind='stable')
        mapping = np.empty(n_train_categories + 1, dtype=np.int64)
        mapping[order] = np.arange(n_train_categories, dtype=np.int64)
        mapping[-1] = n_train_categories
        mappings.append(mapping)
        for part, values in encoded.items():
            col = values[:, column]
            unknown = col >= n_train_categories
            safe_col = np.where(unknown, n_train_categories, col)
            transformed[part][:, column] = mapping[safe_col]
    return (transformed, {'mappings': mappings, 'unknown_value': unknown_value})
