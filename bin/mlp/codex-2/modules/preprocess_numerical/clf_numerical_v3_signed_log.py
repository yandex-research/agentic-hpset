# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np


def numerical_preprocess_v3(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (
            None,
            {'center': None, 'log_center': None, 'scale': None, 'keep_mask': None},
        )
    train = np.asarray(x_num['train'], dtype=np.float32)
    center = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0).astype(
        np.float32
    )
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    imputed_train = np.where(np.isfinite(train), train, center)
    logged_train = np.sign(imputed_train) * np.log1p(np.abs(imputed_train))
    log_center = np.median(logged_train, axis=0).astype(np.float32)
    q25, q75 = np.quantile(logged_train, [0.25, 0.75], axis=0)
    scale = (q75 - q25).astype(np.float32)
    fallback = np.std(logged_train, axis=0).astype(np.float32)
    scale = np.where(scale > 1e-06, scale, fallback)
    scale = np.where(scale > 1e-06, scale, 1.0).astype(np.float32)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        values = np.asarray(values, dtype=np.float32)
        imputed = np.where(np.isfinite(values), values, center)
        logged = np.sign(imputed) * np.log1p(np.abs(imputed))
        transformed[part] = np.clip((logged - log_center) / scale, -8.0, 8.0).astype(
            np.float32
        )
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {
                'center': center,
                'log_center': log_center,
                'scale': scale,
                'keep_mask': keep_mask,
            },
        )
    return (
        transformed,
        {
            'center': center,
            'log_center': log_center,
            'scale': scale,
            'keep_mask': keep_mask,
        },
    )
