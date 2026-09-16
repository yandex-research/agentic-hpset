# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing
from ...core import bounded_numerical_parts


def numerical_preprocess_v1(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Yeo-Johnson PowerTransformer on numeric features.

    Handles negative values and skewed distributions (better than QuantileTransformer
    for low-sample regimes). Falls back to standardization for failure cases.
    """
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v1 requires a seed in config.')
    bounded, support = bounded_numerical_parts(x_num)
    x_num_train = bounded['train']
    transformer = sklearn.preprocessing.PowerTransformer(
        method='yeo-johnson', standardize=True
    )
    noisy_train = x_num_train + np.random.RandomState(seed).normal(
        0.0, 1e-05, x_num_train.shape
    ).astype(x_num_train.dtype)
    try:
        transformer.fit(noisy_train)
        transformed = {
            part: transformer.transform(values) for part, values in bounded.items()
        }
    except Exception:
        scaler = sklearn.preprocessing.StandardScaler().fit(noisy_train)
        transformed = {
            part: scaler.transform(values) for part, values in bounded.items()
        }
        transformer = scaler
    transformed = {
        part: np.nan_to_num(values).astype(np.float32)
        for part, values in transformed.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {'transformer': transformer, 'keep_mask': keep_mask, **support},
        )
    return (
        transformed,
        {'transformer': transformer, 'keep_mask': keep_mask, **support},
    )
