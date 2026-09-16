# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
from ...core import ordinal_encode_categories


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Frequency encoding: keep ordinal codes but order them by training frequency
    (most-frequent category becomes 0, ascending). The output remains integer-coded
    so the downstream cardinality-aware embeddings still work."""
    if x_cat is None:
        return (None, {'encoder': None, 'remap': None, 'unknown_value': None})
    encoded, maps, unknown_codes = ordinal_encode_categories(x_cat)
    n_columns = encoded['train'].shape[1]
    remap_per_column: list[dict[int, int]] = []
    for column in range(n_columns):
        codes, counts = np.unique(encoded['train'][:, column], return_counts=True)
        order = np.argsort(-counts, kind='stable')
        remap = {int(code): rank for rank, code in enumerate(codes[order].tolist())}
        remap_per_column.append(remap)
    for part in encoded:
        for column in range(n_columns):
            remap = remap_per_column[column]
            unknown_code = int(unknown_codes[column])
            applied = np.array(
                [remap.get(int(code), len(remap)) for code in encoded[part][:, column]],
                dtype=np.int64,
            )
            applied[encoded[part][:, column] == unknown_code] = len(remap)
            encoded[part][:, column] = applied
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
