# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np
from ...core import ordinal_encode_categories


def categorical_preprocess_v2(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Group-size encoding: re-rank categories by ascending group size in train.
    Smallest group gets code 0, largest gets the highest code. Acts as a sort of
    'rarity index' that downstream cardinality-aware embeddings can still consume."""
    if x_cat is None:
        return (
            None,
            {'encoder': None, 'remap_per_column': None, 'unknown_value': None},
        )
    encoded, maps, unknown_codes = ordinal_encode_categories(x_cat)
    n_columns = encoded['train'].shape[1]
    remap_per_column: list[dict[int, int]] = []
    for column in range(n_columns):
        codes, counts = np.unique(encoded['train'][:, column], return_counts=True)
        order = np.argsort(counts, kind='stable')
        remap_per_column.append(
            {int(code): rank for rank, code in enumerate(codes[order].tolist())}
        )
    for part in encoded:
        for column in range(n_columns):
            remap = remap_per_column[column]
            unknown_code = int(unknown_codes[column])
            new_codes = np.array(
                [remap.get(int(code), len(remap)) for code in encoded[part][:, column]],
                dtype=np.int64,
            )
            new_codes[encoded[part][:, column] == unknown_code] = len(remap)
            encoded[part][:, column] = new_codes
    return (
        encoded,
        {
            'encoder': maps,
            'remap_per_column': remap_per_column,
            'unknown_value': np.asarray(
                [len(remap) for remap in remap_per_column], dtype=np.int64
            ),
        },
    )
