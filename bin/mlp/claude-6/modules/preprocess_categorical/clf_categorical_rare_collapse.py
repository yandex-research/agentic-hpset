# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
from ...core import ordinal_encode_categories


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'encoder': None, 'unknown_value': None})
    config = config or {}
    min_count = int(config.get('rare_min_count', 5))
    encoded, maps, source_unknown = ordinal_encode_categories(x_cat)
    rare_sets: list[set[int]] = []
    unknown_codes: list[int] = []
    for column, mapping in enumerate(maps):
        counts = np.bincount(encoded['train'][:, column], minlength=len(mapping))
        rare = {int(code) for code in np.flatnonzero(counts < min_count)}
        rare_sets.append(rare)
        rare_code = len(mapping)
        unknown_code = rare_code + int(bool(rare))
        unknown_codes.append(unknown_code)
        for part, values in encoded.items():
            column_values = values[:, column]
            is_unknown = column_values == int(source_unknown[column])
            is_rare = np.isin(column_values, list(rare))
            if rare:
                column_values[is_rare] = rare_code
            column_values[is_unknown] = unknown_code
    return (
        encoded,
        {
            'encoder': maps,
            'unknown_value': np.asarray(unknown_codes, dtype=np.int64),
            'rare_sets': rare_sets,
        },
    )
