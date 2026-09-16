# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _non_constant_mask(values: np.ndarray) -> np.ndarray:
    return np.array(
        [len(np.unique(column[~np.isnan(column)])) > 1 for column in values.T],
        dtype=bool,
    )


def numerical_preprocess_v1(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'imputer': None, 'scaler': None, 'keep_mask': None})
    keep_mask = _non_constant_mask(x_num['train'])
    if keep_mask.sum() == 0:
        return (None, {'imputer': None, 'scaler': None, 'keep_mask': keep_mask})
    imputer = sklearn.impute.SimpleImputer(strategy='median')
    scaler = sklearn.preprocessing.StandardScaler()
    train = imputer.fit_transform(x_num['train'][:, keep_mask])
    scaler.fit(train)
    transformed = {
        part: scaler.transform(imputer.transform(values[:, keep_mask]))
        for part, values in x_num.items()
    }
    transformed = {
        part: np.nan_to_num(values).astype(np.float32)
        for part, values in transformed.items()
    }
    return (transformed, {'imputer': imputer, 'scaler': scaler, 'keep_mask': keep_mask})
