# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np
from ...core import ordinal_encode_categories


def categorical_preprocess_v2(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Replace each category with a log2 frequency bucket.

    The cardinality of the output is the number of distinct log2 buckets, which
    is much smaller than the raw cardinality for high-card columns. Unknowns map
    to bucket 0.
    """
    if x_cat is None:
        return (None, {'encoder': None, 'unknown_value': None, 'count_tables': None})
    encoded, maps, unknown_codes = ordinal_encode_categories(x_cat)
    n_features = encoded['train'].shape[1]
    count_tables: list[np.ndarray] = []
    for col in range(n_features):
        train_col = encoded['train'][:, col]
        max_idx = int(train_col.max()) if train_col.size else -1
        counts = np.bincount(np.maximum(train_col, 0), minlength=max_idx + 1)
        count_tables.append(counts.astype(np.int64))
    bucketed = {part: np.zeros_like(values) for part, values in encoded.items()}
    for col in range(n_features):
        counts = count_tables[col]
        log_buckets = 1 + np.floor(
            np.log2(np.maximum(counts.astype(np.float64), 1.0))
        ).astype(np.int64)
        for part in encoded:
            col_vals = encoded[part][:, col]
            mask_unknown = col_vals == int(unknown_codes[col])
            safe_idx = np.where(mask_unknown, 0, col_vals).clip(0, counts.size - 1)
            bucketed[part][:, col] = np.where(mask_unknown, 0, log_buckets[safe_idx])
    identity_and_frequency = {
        part: np.concatenate([encoded[part], bucketed[part]], axis=1)
        for part in encoded
    }
    return (
        identity_and_frequency,
        {
            'encoder': maps,
            'unknown_value': unknown_codes,
            'count_tables': count_tables,
            'identity_features': n_features,
        },
    )
