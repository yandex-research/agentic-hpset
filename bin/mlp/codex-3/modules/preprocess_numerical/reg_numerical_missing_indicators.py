# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v5``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v5(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (
            None,
            {
                'transformer': None,
                'keep_mask': None,
                'missing_mask': None,
                'medians': None,
            },
        )
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v5 requires a seed in config.')
    train = x_num['train'].astype(np.float32, copy=False)
    medians = np.nanmedian(train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0).astype(np.float32)
    missing_mask = np.isnan(train).any(axis=0)
    filled: dict[str, np.ndarray] = {}
    indicators: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        values = values.astype(np.float32, copy=True)
        indicators[part] = np.isnan(values[:, missing_mask]).astype(np.float32)
        mask = np.isnan(values)
        if mask.any():
            values[mask] = np.take(medians, np.where(mask)[1])
        filled[part] = values
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(train.shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    noise = np.random.RandomState(seed + 43).normal(0.0, 1e-05, filled['train'].shape)
    transformer.fit(filled['train'] + noise.astype(filled['train'].dtype))
    base = {
        part: np.clip(np.nan_to_num(transformer.transform(values)), -5.0, 5.0).astype(
            np.float32
        )
        for part, values in filled.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in base['train'].T], dtype=bool
    )
    transformed: dict[str, np.ndarray] = {}
    for part, values in base.items():
        pieces = [values[:, keep_mask]]
        if indicators[part].shape[1]:
            pieces.append(indicators[part])
        transformed[part] = np.column_stack(pieces).astype(np.float32)
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {
                'transformer': transformer,
                'keep_mask': keep_mask,
                'missing_mask': missing_mask,
                'medians': medians,
            },
        )
    return (
        transformed,
        {
            'transformer': transformer,
            'keep_mask': keep_mask,
            'missing_mask': missing_mask,
            'medians': medians,
        },
    )
