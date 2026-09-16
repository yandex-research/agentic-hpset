# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _fill_values(x_train: np.ndarray) -> np.ndarray:
    fill = np.nanmedian(x_train, axis=0)
    return np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)


def numerical_preprocess_v3(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v3 requires a seed in config.')
    x_train = x_num['train']
    fill = _fill_values(x_train)
    train_filled = np.where(np.isnan(x_train), fill, x_train).astype(np.float32)
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_train.shape[0] // 20, 2000), 10),
        output_distribution='uniform',
        subsample=1000000000,
        random_state=seed,
    )
    transformer.fit(train_filled)
    transformed = {}
    for part, values in x_num.items():
        filled = np.where(np.isnan(values), fill, values).astype(np.float32)
        ranks = transformer.transform(filled)
        transformed[part] = ((ranks - 0.5) * 2.0).astype(np.float32)
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'transformer': transformer, 'keep_mask': keep_mask})
    return (transformed, {'transformer': transformer, 'keep_mask': keep_mask})
