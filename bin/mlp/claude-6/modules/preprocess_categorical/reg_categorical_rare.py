# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
from ...core import ordinal_encode_categories


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Ordinal encode, then collapse train-rare categories into a shared `rare` bucket.

    Threshold: a category is `rare` if its train frequency is below max(2, 0.5%).
    Unknown / val-only categories also fall into the `rare` bucket.
    """
    if x_cat is None:
        return (None, {'encoder': None, 'unknown_value': None, 'rare_thresholds': None})
    config = config or {}
    rare_threshold_frac = float(config.get('rare_threshold_frac', 0.005))
    encoded, maps, source_unknown = ordinal_encode_categories(x_cat)
    n_train = encoded['train'].shape[0]
    n_features = encoded['train'].shape[1]
    rare_thresholds: list[int] = []
    unknown_codes: list[int] = []
    for col in range(n_features):
        train_col = encoded['train'][:, col]
        unique, counts = np.unique(train_col, return_counts=True)
        threshold = max(2, int(rare_threshold_frac * n_train))
        rare_categories = set((int(u) for u, c in zip(unique, counts) if c < threshold))
        rare_id = len(maps[col])
        unknown_id = rare_id + int(bool(rare_categories))
        rare_thresholds.append(threshold)
        unknown_codes.append(unknown_id)
        for part in encoded:
            col_vals = encoded[part][:, col]
            mask_rare = np.isin(col_vals, list(rare_categories))
            mask_unknown = col_vals == int(source_unknown[col])
            if rare_categories:
                col_vals[mask_rare] = rare_id
            col_vals[mask_unknown] = unknown_id
            encoded[part][:, col] = col_vals
    return (
        encoded,
        {
            'encoder': maps,
            'unknown_value': np.asarray(unknown_codes, dtype=np.int64),
            'rare_thresholds': rare_thresholds,
        },
    )
