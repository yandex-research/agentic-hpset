# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _fill_nan_with_train_median(
    x_num: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    medians = np.nanmedian(x_num['train'], axis=0)
    medians = np.nan_to_num(medians, nan=0.0).astype(np.float32)
    filled: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        values = values.astype(np.float32, copy=True)
        mask = np.isnan(values)
        if mask.any():
            values[mask] = np.take(medians, np.where(mask)[1])
        filled[part] = values
    return (filled, medians)


def numerical_preprocess_v1(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None, 'medians': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v1 requires a seed in config.')
    filled, medians = _fill_nan_with_train_median(x_num)
    x_train = filled['train']
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_train.shape[0] // 20, 1000), 10),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    noise = np.random.RandomState(seed + 17).normal(0.0, 1e-05, x_train.shape)
    transformer.fit(x_train + noise.astype(x_train.dtype))
    transformed = {
        part: np.clip(np.nan_to_num(transformer.transform(values)), -5.0, 5.0).astype(
            np.float32
        )
        for part, values in filled.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {'transformer': transformer, 'keep_mask': keep_mask, 'medians': medians},
        )
    return (
        transformed,
        {'transformer': transformer, 'keep_mask': keep_mask, 'medians': medians},
    )
