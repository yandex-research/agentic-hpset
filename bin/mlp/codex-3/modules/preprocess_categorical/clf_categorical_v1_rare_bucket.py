# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'mappings': None, 'rare_thresholds': None})
    train = x_cat['train']
    mappings: list[dict[object, int]] = []
    rare_values: list[int] = []
    for column in range(train.shape[1]):
        values, counts = np.unique(train[:, column], return_counts=True)
        threshold = max(2, int(np.ceil(0.005 * train.shape[0])))
        kept = values[counts >= threshold]
        mapping = {value: idx for idx, value in enumerate(kept.tolist())}
        mappings.append(mapping)
        rare_values.append(len(mapping))
    encoded: dict[str, np.ndarray] = {}
    for part, values in x_cat.items():
        out = np.empty(values.shape, dtype=np.int64)
        for column, mapping in enumerate(mappings):
            rare_value = rare_values[column]
            out[:, column] = [
                mapping.get(value, rare_value) for value in values[:, column]
            ]
        encoded[part] = out
    return (encoded, {'mappings': mappings, 'rare_values': rare_values})
