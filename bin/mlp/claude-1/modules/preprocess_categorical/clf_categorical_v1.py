# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'encoder': None, 'unknown_value': None, 'rare_maps': None})
    config = config or {}
    rare_threshold = int(config.get('rare_threshold', 10))
    sentinel = np.iinfo(np.int64).max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=sentinel, dtype=np.int64
    ).fit(x_cat['train'])
    encoded = {
        part: encoder.transform(values).astype(np.int64)
        for part, values in x_cat.items()
    }
    n_features = encoded['train'].shape[1]
    rare_maps: list[dict[str, Any]] = []
    for j in range(n_features):
        col = encoded['train'][:, j]
        unique, counts = np.unique(col, return_counts=True)
        keep_sorted = sorted(
            (
                int(v)
                for v, c in zip(unique.tolist(), counts.tolist())
                if c >= rare_threshold
            )
        )
        mapping = {int(v): idx for idx, v in enumerate(keep_sorted)}
        rare_id = len(mapping)
        unknown_id = rare_id + 1
        rare_maps.append({'map': mapping, 'rare': rare_id, 'unknown': unknown_id})
    new_encoded: dict[str, np.ndarray] = {}
    for part, col_arr in encoded.items():
        new_arr = np.empty_like(col_arr)
        for j in range(n_features):
            info = rare_maps[j]
            mapping = info['map']
            rare = info['rare']
            unknown = info['unknown']
            col = col_arr[:, j]
            out = np.full(col.shape, rare, dtype=np.int64)
            out[col == sentinel] = unknown
            for original, new_id in mapping.items():
                out[col == original] = new_id
            new_arr[:, j] = out
        new_encoded[part] = new_arr
    return (
        new_encoded,
        {'encoder': encoder, 'unknown_value': sentinel, 'rare_maps': rare_maps},
    )
