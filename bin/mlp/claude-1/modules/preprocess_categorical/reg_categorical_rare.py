# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_rare``."""

from __future__ import annotations
from typing import Any
import numpy as np


def categorical_preprocess_rare(
    x_cat: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'category_maps': None})
    config = config or {}
    min_count = int(config.get('rare_min_count', 5))
    train = x_cat['train']
    n_cols = train.shape[1]
    category_maps: list[dict[Any, int]] = []
    rare_codes: list[int] = []
    encoded: dict[str, np.ndarray] = {
        part: np.zeros_like(values, dtype=np.int64) for part, values in x_cat.items()
    }
    for col in range(n_cols):
        unique, counts = np.unique(train[:, col], return_counts=True)
        common = unique[counts >= min_count]
        mapping: dict[Any, int] = {
            value: idx for idx, value in enumerate(common.tolist())
        }
        rare_code = len(mapping)
        unknown_code = rare_code + 1
        category_maps.append(mapping)
        rare_codes.append(rare_code)
        for part, values in x_cat.items():
            col_vals = values[:, col]
            mapped = np.full(col_vals.shape[0], unknown_code, dtype=np.int64)
            for value, idx in mapping.items():
                mapped[col_vals == value] = idx
            in_train_mask = np.isin(col_vals, unique)
            mapped[in_train_mask & (mapped == unknown_code)] = rare_code
            encoded[part][:, col] = mapped
    return (encoded, {'category_maps': category_maps, 'rare_codes': rare_codes})
