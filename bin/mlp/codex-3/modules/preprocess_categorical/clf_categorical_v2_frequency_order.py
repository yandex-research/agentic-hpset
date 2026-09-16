# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np


def categorical_preprocess_v2(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'mappings': None})
    train = x_cat['train']
    mappings: list[dict[object, int]] = []
    unknown_values: list[int] = []
    for column in range(train.shape[1]):
        values, counts = np.unique(train[:, column], return_counts=True)
        order = np.lexsort((values.astype(str), -counts))
        ordered = values[order]
        mapping = {value: idx for idx, value in enumerate(ordered.tolist())}
        mappings.append(mapping)
        unknown_values.append(len(mapping))
    encoded: dict[str, np.ndarray] = {}
    for part, values in x_cat.items():
        out = np.empty(values.shape, dtype=np.int64)
        for column, mapping in enumerate(mappings):
            unknown = unknown_values[column]
            out[:, column] = [
                mapping.get(value, unknown) for value in values[:, column]
            ]
        encoded[part] = out
    return (encoded, {'mappings': mappings, 'unknown_values': unknown_values})
