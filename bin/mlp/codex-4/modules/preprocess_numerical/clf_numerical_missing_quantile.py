# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _fill_values(x_train: np.ndarray) -> np.ndarray:
    fill = np.nanmedian(x_train, axis=0)
    return np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)


def _remove_constant(
    transformed: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray] | None, np.ndarray]:
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, keep_mask)
    return (transformed, keep_mask)


def numerical_preprocess_v1(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None, 'missing_mask': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v1 requires a seed in config.')
    x_train = x_num['train']
    fill = _fill_values(x_train)
    missing_mask = np.isnan(x_train).any(axis=0)
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_train.shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    train_filled = np.where(np.isnan(x_train), fill, x_train).astype(np.float32)
    noisy_train = train_filled + np.random.RandomState(seed).normal(
        0.0, 1e-05, train_filled.shape
    ).astype(np.float32)
    transformer.fit(noisy_train)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        filled = np.where(np.isnan(values), fill, values).astype(np.float32)
        main = transformer.transform(filled)
        main = np.clip(np.nan_to_num(main), -5.0, 5.0).astype(np.float32)
        if missing_mask.any():
            indicators = np.isnan(values[:, missing_mask]).astype(np.float32)
            main = np.concatenate([main, indicators], axis=1)
        transformed[part] = main
    transformed, keep_mask = _remove_constant(transformed)
    return (
        transformed,
        {
            'transformer': transformer,
            'keep_mask': keep_mask,
            'missing_mask': missing_mask,
        },
    )
