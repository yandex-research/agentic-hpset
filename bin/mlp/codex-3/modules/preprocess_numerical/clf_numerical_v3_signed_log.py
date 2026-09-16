# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def numerical_preprocess_v3(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'median': None, 'iqr': None, 'keep_mask': None})
    train = x_num['train'].astype(np.float32, copy=False)
    raw_median = np.nanmedian(train, axis=0).astype(np.float32)
    raw_median = np.nan_to_num(raw_median, nan=0.0).astype(np.float32)
    log_train = _signed_log(np.where(np.isnan(train), raw_median, train)).astype(
        np.float32
    )
    median = np.median(log_train, axis=0).astype(np.float32)
    q25 = np.percentile(log_train, 25, axis=0).astype(np.float32)
    q75 = np.percentile(log_train, 75, axis=0).astype(np.float32)
    iqr = q75 - q25
    iqr = np.where(np.isfinite(iqr) & (np.abs(iqr) > 1e-06), iqr, 1.0).astype(
        np.float32
    )
    transformed = {}
    for part, values in x_num.items():
        values = values.astype(np.float32, copy=False)
        logged = _signed_log(np.where(np.isnan(values), raw_median, values)).astype(
            np.float32
        )
        transformed[part] = np.clip((logged - median) / iqr, -8.0, 8.0).astype(
            np.float32
        )
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'median': median, 'iqr': iqr, 'keep_mask': keep_mask})
    return (transformed, {'median': median, 'iqr': iqr, 'keep_mask': keep_mask})
