# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def categorical_preprocess_v3(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'encoder': None, 'column_maps': None})
    unknown_value = np.iinfo(np.int64).max - 17
    missing_value = np.iinfo(np.int64).max - 19
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=unknown_value,
        encoded_missing_value=missing_value,
        dtype=np.int64,
    ).fit(x_cat['train'])
    encoded = {
        part: encoder.transform(values).astype(np.int64)
        for part, values in x_cat.items()
    }
    n_rows = encoded['train'].shape[0]
    min_count = max(2, int(round(0.005 * n_rows)))
    column_maps: list[np.ndarray] = []
    for column in range(encoded['train'].shape[1]):
        train_col = encoded['train'][:, column]
        regular = train_col[(train_col != unknown_value) & (train_col != missing_value)]
        max_code = int(regular.max()) if regular.size else -1
        counts = (
            np.bincount(regular, minlength=max_code + 1)
            if max_code >= 0
            else np.array([])
        )
        frequent = counts >= min_count
        mapping = np.full(max_code + 1, int(frequent.sum()), dtype=np.int64)
        mapping[frequent] = np.arange(int(frequent.sum()), dtype=np.int64)
        column_maps.append(mapping)
    for part, values in encoded.items():
        out = np.empty_like(values)
        for column, mapping in enumerate(column_maps):
            rare_id = len(mapping) if mapping.size == 0 else int(mapping.max()) + 1
            missing_id = rare_id + 1
            col = values[:, column]
            mapped = np.full(col.shape, rare_id, dtype=np.int64)
            known = (col >= 0) & (col < len(mapping))
            mapped[known] = mapping[col[known]]
            mapped[col == missing_value] = missing_id
            out[:, column] = mapped
        encoded[part] = out
    return (
        encoded,
        {
            'encoder': encoder,
            'column_maps': column_maps,
            'min_count': min_count,
            'unknown_value': unknown_value,
            'missing_value': missing_value,
        },
    )
