# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _non_constant_mask(values: np.ndarray) -> np.ndarray:
    return np.array(
        [len(np.unique(column[~np.isnan(column)])) > 1 for column in values.T],
        dtype=bool,
    )


def numerical_preprocess_v2(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'imputer': None, 'scaler': None, 'keep_mask': None})
    keep_mask = _non_constant_mask(x_num['train'])
    if keep_mask.sum() == 0:
        return (None, {'imputer': None, 'scaler': None, 'keep_mask': keep_mask})
    imputer = sklearn.impute.SimpleImputer(strategy='median')
    train = imputer.fit_transform(x_num['train'][:, keep_mask])
    low, high = np.quantile(train, [0.01, 0.99], axis=0)
    train_clipped = np.clip(train, low, high)
    scaler = sklearn.preprocessing.RobustScaler(quantile_range=(10.0, 90.0))
    scaler.fit(train_clipped)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        imputed = imputer.transform(values[:, keep_mask])
        clipped = np.clip(imputed, low, high)
        transformed[part] = scaler.transform(clipped)
    transformed = {
        part: np.nan_to_num(values).astype(np.float32)
        for part, values in transformed.items()
    }
    return (
        transformed,
        {
            'imputer': imputer,
            'scaler': scaler,
            'keep_mask': keep_mask,
            'clip_low': low,
            'clip_high': high,
        },
    )
