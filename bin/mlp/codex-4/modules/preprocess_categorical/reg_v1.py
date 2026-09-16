# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
from ...core import ordinal_encode_categories


def _apply_maps(
    encoded: dict[str, np.ndarray], maps: list[np.ndarray], unknown_codes: np.ndarray
) -> dict[str, np.ndarray]:
    mapped: dict[str, np.ndarray] = {}
    for part, values in encoded.items():
        output = np.empty_like(values, dtype=np.int64)
        for column, mapping in enumerate(maps):
            clipped = np.minimum(values[:, column], unknown_codes[column])
            output[:, column] = mapping[clipped]
        mapped[part] = output
    return mapped


def _ordinal_encode(
    x_cat: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict[str, int]], np.ndarray]:
    return ordinal_encode_categories(x_cat)


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Group low-count categories into a shared rare bucket per feature."""
    if x_cat is None:
        return (None, {'variant': 'rare_bucket', 'encoder': None})
    encoded, encoder, source_unknown = _ordinal_encode(x_cat)
    maps: list[np.ndarray] = []
    unknown_codes: list[int] = []
    for column in range(encoded['train'].shape[1]):
        max_train = int(encoded['train'][:, column].max())
        unknown_code = int(source_unknown[column])
        counts = np.bincount(encoded['train'][:, column], minlength=unknown_code + 1)
        min_count = max(2, int(np.sqrt(encoded['train'].shape[0]) // 4))
        frequent = counts[: max_train + 1] >= min_count
        mapping = np.zeros(unknown_code + 1, dtype=np.int64)
        next_code = 1
        for code, keep in enumerate(frequent):
            if keep:
                mapping[code] = next_code
                next_code += 1
        mapping[unknown_code] = next_code
        maps.append(mapping)
        unknown_codes.append(unknown_code)
    return (
        _apply_maps(encoded, maps, np.asarray(unknown_codes)),
        {
            'variant': 'rare_bucket',
            'encoder': encoder,
            'maps': maps,
            'unknown_codes': np.asarray(
                [mapping[-1] for mapping in maps], dtype=np.int64
            ),
        },
    )
