# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _fill_values(x_train: np.ndarray) -> np.ndarray:
    fill = np.nanmedian(x_train, axis=0)
    return np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)


def numerical_preprocess_v2(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'keep_mask': None, 'missing_mask': None})
    x_train = x_num['train']
    fill = _fill_values(x_train)
    train_filled = np.where(np.isnan(x_train), fill, x_train).astype(np.float32)
    lower = np.nanquantile(train_filled, 0.005, axis=0).astype(np.float32)
    upper = np.nanquantile(train_filled, 0.995, axis=0).astype(np.float32)
    median = np.nanmedian(train_filled, axis=0).astype(np.float32)
    q25 = np.nanquantile(train_filled, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(train_filled, 0.75, axis=0).astype(np.float32)
    scale = (q75 - q25).astype(np.float32)
    fallback = np.nanstd(train_filled, axis=0).astype(np.float32)
    scale = np.where(scale > 1e-06, scale, fallback)
    scale = np.where(scale > 1e-06, scale, 1.0).astype(np.float32)
    missing_mask = np.isnan(x_train).any(axis=0)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        filled = np.where(np.isnan(values), fill, values).astype(np.float32)
        clipped = np.clip(filled, lower, upper)
        main = np.clip((clipped - median) / scale, -8.0, 8.0).astype(np.float32)
        if missing_mask.any():
            main = np.concatenate(
                [main, np.isnan(values[:, missing_mask]).astype(np.float32)], axis=1
            )
        transformed[part] = main
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'keep_mask': keep_mask, 'missing_mask': missing_mask})
    return (transformed, {'keep_mask': keep_mask, 'missing_mask': missing_mask})
