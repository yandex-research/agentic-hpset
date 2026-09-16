# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
from collections import Counter
import numpy as np
from ...core import categorical_key


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'maps': None, 'rare_thresholds': None})
    n_features = x_cat['train'].shape[1]
    maps: list[dict[str, int]] = []
    train_keys: list[set[str]] = []
    rare_thresholds: list[int] = []
    for column in range(n_features):
        keys = [categorical_key(value) for value in x_cat['train'][:, column]]
        counts = Counter(keys)
        threshold = max(2, int(round(0.0025 * len(keys))))
        common = [
            key
            for key, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if count >= threshold
        ]
        maps.append({key: index for index, key in enumerate(common)})
        train_keys.append(set(keys))
        rare_thresholds.append(threshold)
    encoded = {}
    for part, values in x_cat.items():
        output = np.empty(values.shape, dtype=np.int64)
        for column, mapping in enumerate(maps):
            rare_index = len(mapping)
            has_rare = bool(train_keys[column] - mapping.keys())
            unknown_index = rare_index + int(has_rare)
            output[:, column] = [
                mapping.get(
                    categorical_key(value),
                    rare_index
                    if categorical_key(value) in train_keys[column]
                    else unknown_index,
                )
                for value in values[:, column]
            ]
        encoded[part] = output
    return (
        encoded,
        {
            'maps': maps,
            'rare_thresholds': rare_thresholds,
            'unknown_codes': np.asarray(
                [
                    len(mapping) + int(bool(train_keys[column] - mapping.keys()))
                    for column, mapping in enumerate(maps)
                ],
                dtype=np.int64,
            ),
        },
    )
