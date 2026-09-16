# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
from collections import Counter
import numpy as np
from ...core import categorical_key


def categorical_preprocess_v2(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'maps': None})
    maps: list[dict[str, int]] = []
    for column in range(x_cat['train'].shape[1]):
        counts = Counter(
            (categorical_key(value) for value in x_cat['train'][:, column])
        )
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        maps.append({key: index for index, (key, _count) in enumerate(ordered)})
    encoded = {}
    for part, values in x_cat.items():
        output = np.empty(values.shape, dtype=np.int64)
        for column, mapping in enumerate(maps):
            unknown_index = len(mapping)
            output[:, column] = [
                mapping.get(categorical_key(value), unknown_index)
                for value in values[:, column]
            ]
        encoded[part] = output
    return (encoded, {'maps': maps})
