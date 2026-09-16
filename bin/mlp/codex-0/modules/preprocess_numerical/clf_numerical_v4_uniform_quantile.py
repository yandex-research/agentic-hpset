# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v4``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _non_constant_mask(values: np.ndarray) -> np.ndarray:
    return np.array(
        [len(np.unique(column[~np.isnan(column)])) > 1 for column in values.T],
        dtype=bool,
    )


def numerical_preprocess_v4(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'imputer': None, 'transformer': None, 'keep_mask': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v4 requires a seed in config.')
    keep_mask = _non_constant_mask(x_num['train'])
    if keep_mask.sum() == 0:
        return (None, {'imputer': None, 'transformer': None, 'keep_mask': keep_mask})
    imputer = sklearn.impute.SimpleImputer(strategy='median')
    train = imputer.fit_transform(x_num['train'][:, keep_mask])
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(train.shape[0] // 20, 1000), 10),
        output_distribution='uniform',
        subsample=1000000000,
        random_state=seed,
    )
    noisy_train = train + np.random.RandomState(seed).normal(0.0, 1e-05, train.shape)
    transformer.fit(noisy_train)
    transformed = {
        part: transformer.transform(imputer.transform(values[:, keep_mask])) * 2.0 - 1.0
        for part, values in x_num.items()
    }
    transformed = {
        part: np.nan_to_num(values).astype(np.float32)
        for part, values in transformed.items()
    }
    return (
        transformed,
        {'imputer': imputer, 'transformer': transformer, 'keep_mask': keep_mask},
    )
