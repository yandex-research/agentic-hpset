# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'encoder': None, 'mappings': None, 'unknown_value': None})
    unknown_value = np.iinfo(np.int64).max - 7
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=unknown_value, dtype=np.int64
    ).fit(x_cat['train'])
    encoded = {
        part: encoder.transform(values).astype(np.int64)
        for part, values in x_cat.items()
    }
    mappings: list[np.ndarray] = []
    for column in range(encoded['train'].shape[1]):
        train_column = encoded['train'][:, column]
        cardinality = int(train_column.max()) + 1 if train_column.size else 0
        counts = np.bincount(train_column, minlength=cardinality)
        order = np.argsort(-counts, kind='stable')
        mapping = np.empty(cardinality + 1, dtype=np.int64)
        mapping[order] = np.arange(cardinality, dtype=np.int64)
        mapping[-1] = cardinality
        mappings.append(mapping)
        for part in encoded:
            column_values = encoded[part][:, column]
            unknown_mask = column_values == unknown_value
            safe_values = np.where(unknown_mask, cardinality, column_values)
            encoded[part][:, column] = mapping[safe_values]
    return (
        encoded,
        {'encoder': encoder, 'mappings': mappings, 'unknown_value': unknown_value},
    )
