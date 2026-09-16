# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v2(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'scaler': None, 'keep_mask': None})
    train = x_num['train'].astype(np.float32, copy=False)
    medians = np.nanmedian(train, axis=0).astype(np.float32)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    imputed = {
        part: np.where(np.isnan(values), medians, values).astype(np.float32)
        for part, values in x_num.items()
    }
    low = np.nanpercentile(imputed['train'], 1.0, axis=0).astype(np.float32)
    high = np.nanpercentile(imputed['train'], 99.0, axis=0).astype(np.float32)
    clipped = {part: np.clip(values, low, high) for part, values in imputed.items()}
    keep_mask = np.nanstd(clipped['train'], axis=0) > 0.0
    if not keep_mask.any():
        return (
            None,
            {
                'scaler': None,
                'keep_mask': keep_mask,
                'medians': medians,
                'low': low,
                'high': high,
            },
        )
    scaler = sklearn.preprocessing.StandardScaler().fit(clipped['train'][:, keep_mask])
    transformed = {
        part: scaler.transform(values[:, keep_mask]).astype(np.float32)
        for part, values in clipped.items()
    }
    transformed = {
        part: np.nan_to_num(values, nan=0.0, posinf=5.0, neginf=-5.0)
        for part, values in transformed.items()
    }
    return (
        transformed,
        {
            'scaler': scaler,
            'keep_mask': keep_mask,
            'medians': medians,
            'low': low,
            'high': high,
        },
    )
