# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np


def numerical_preprocess_v3(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'center': None, 'scale': None, 'keep_mask': None})
    train = x_num['train'].astype(np.float32, copy=False)
    center = np.nanmedian(train, axis=0)
    q25 = np.nanpercentile(train, 25.0, axis=0)
    q75 = np.nanpercentile(train, 75.0, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(train, axis=0)
    scale = np.where(scale > 1e-06, scale, fallback)
    center = np.nan_to_num(center, nan=0.0).astype(np.float32)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    scale = np.where(scale > 1e-06, scale, 1.0).astype(np.float32)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        values = values.astype(np.float32, copy=True)
        mask = np.isnan(values)
        if mask.any():
            values[mask] = np.take(center, np.where(mask)[1])
        values = np.clip((values - center) / scale, -10.0, 10.0).astype(np.float32)
        transformed[part] = values
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'center': center, 'scale': scale, 'keep_mask': keep_mask})
    return (transformed, {'center': center, 'scale': scale, 'keep_mask': keep_mask})
