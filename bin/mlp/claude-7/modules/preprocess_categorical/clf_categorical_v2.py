# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np


def categorical_preprocess_v2(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Frequency-rank bucketization of categorical features.

    For each categorical column, each unique value is replaced by an integer
    bucket id corresponding to its training-set frequency rank (binned into
    n_buckets equal-frequency buckets). This collapses high-cardinality
    columns to a fixed small cardinality, encourages the embedding to learn
    semantics tied to popularity (a strong signal in many tabular tasks), and
    delivers a compact, low-variance categorical representation.
    """
    if x_cat is None:
        return (None, {'freq_maps': None, 'fallback_bucket': None, 'n_buckets': None})
    n_buckets = 16
    n_columns = x_cat['train'].shape[1]
    freq_maps: list[dict[object, int]] = []
    for col in range(n_columns):
        values, counts = np.unique(x_cat['train'][:, col], return_counts=True)
        order = np.argsort(-counts, kind='stable')
        values_sorted = values[order]
        bucket_assignments = np.minimum(
            np.arange(values_sorted.size) * n_buckets // max(values_sorted.size, 1),
            n_buckets - 1,
        ).astype(np.int64)
        freq_maps.append(
            {
                value: int(bucket)
                for value, bucket in zip(
                    values_sorted.tolist(), bucket_assignments.tolist()
                )
            }
        )
    fallback_bucket = n_buckets
    encoded: dict[str, np.ndarray] = {}
    for part, values in x_cat.items():
        out = np.empty(values.shape, dtype=np.int64)
        for col in range(n_columns):
            mapper = freq_maps[col]
            for row, item in enumerate(values[:, col].tolist()):
                out[row, col] = mapper.get(item, fallback_bucket)
        encoded[part] = out
    return (
        encoded,
        {
            'freq_maps': freq_maps,
            'fallback_bucket': fallback_bucket,
            'n_buckets': n_buckets,
        },
    )
