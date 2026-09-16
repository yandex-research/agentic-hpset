# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np


def categorical_preprocess_v3(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'mappings': None, 'caps': None})
    train = x_cat['train']
    mappings: list[dict[object, int]] = []
    caps: list[int] = []
    for column in range(train.shape[1]):
        values, counts = np.unique(train[:, column], return_counts=True)
        cap = min(len(values), max(8, min(64, int(np.sqrt(train.shape[0])))))
        order = np.argsort(-counts, kind='stable')[:cap]
        kept = values[order]
        mapping = {value: idx for idx, value in enumerate(kept.tolist())}
        mappings.append(mapping)
        caps.append(cap)
    encoded: dict[str, np.ndarray] = {}
    for part, values in x_cat.items():
        out = np.empty(values.shape, dtype=np.int64)
        for column, mapping in enumerate(mappings):
            other = len(mapping)
            out[:, column] = [mapping.get(value, other) for value in values[:, column]]
        encoded[part] = out
    return (encoded, {'mappings': mappings, 'caps': caps})
