# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v3_frequency``."""

from __future__ import annotations
from typing import Any
import numpy as np


def categorical_preprocess_v3_frequency(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Order categories by descending train-set frequency before assigning ordinal IDs.

    Frequent categories get lower indices, which is a useful prior for one-hot
    or embedding lookups (the most-common bucket is always index 0).
    """
    if x_cat is None:
        return (None, {'orderings': None})
    train_arr = x_cat['train']
    val_arr = x_cat['val']
    test_arr = x_cat['test']
    n_cols = train_arr.shape[1]
    orderings: list[dict[object, int]] = []
    encoded = {
        'train': np.zeros_like(train_arr, dtype=np.int64),
        'val': np.zeros_like(val_arr, dtype=np.int64),
        'test': np.zeros_like(test_arr, dtype=np.int64),
    }
    for col in range(n_cols):
        train_col = train_arr[:, col]
        unique_vals, counts = np.unique(train_col, return_counts=True)
        order = np.argsort(-counts)
        unique_vals = unique_vals[order]
        mapping: dict[object, int] = {v: i for i, v in enumerate(unique_vals.tolist())}
        next_id = len(mapping)
        encoded['train'][:, col] = np.array(
            [mapping[v] for v in train_col], dtype=np.int64
        )
        for part_name, part_arr in (('val', val_arr), ('test', test_arr)):
            mapped = np.empty(part_arr.shape[0], dtype=np.int64)
            for i, v in enumerate(part_arr[:, col]):
                mapped[i] = mapping.get(v, next_id)
            encoded[part_name][:, col] = mapped
        orderings.append(mapping)
    return (encoded, {'orderings': orderings})
