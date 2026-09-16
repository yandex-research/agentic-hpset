# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v4``."""

from __future__ import annotations
from typing import Any
import numpy as np


def numerical_preprocess_v4(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (
            None,
            {
                'mean': None,
                'std': None,
                'square_mean': None,
                'square_std': None,
                'keep_mask': None,
            },
        )
    train = np.asarray(x_num['train'], dtype=np.float32)
    finite_train = np.where(np.isfinite(train), train, np.nan)
    mean = np.nanmean(finite_train, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    imputed_train = np.where(np.isfinite(train), train, mean)
    std = np.std(imputed_train, axis=0).astype(np.float32)
    std = np.where(std > 1e-06, std, 1.0).astype(np.float32)
    z_train = np.clip((imputed_train - mean) / std, -6.0, 6.0)
    square_train = z_train * z_train
    square_mean = square_train.mean(axis=0).astype(np.float32)
    square_std = square_train.std(axis=0).astype(np.float32)
    square_std = np.where(square_std > 1e-06, square_std, 1.0).astype(np.float32)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        values = np.asarray(values, dtype=np.float32)
        imputed = np.where(np.isfinite(values), values, mean)
        z = np.clip((imputed - mean) / std, -6.0, 6.0).astype(np.float32)
        square = (z * z - square_mean) / square_std
        transformed[part] = np.column_stack([z, np.clip(square, -8.0, 8.0)]).astype(
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
                'mean': mean,
                'std': std,
                'square_mean': square_mean,
                'square_std': square_std,
                'keep_mask': keep_mask,
            },
        )
    return (
        transformed,
        {
            'mean': mean,
            'std': std,
            'square_mean': square_mean,
            'square_std': square_std,
            'keep_mask': keep_mask,
        },
    )
