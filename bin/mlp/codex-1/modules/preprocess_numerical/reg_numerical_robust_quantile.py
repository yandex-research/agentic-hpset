# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _finish(
    transformed: dict[str, np.ndarray], artifacts: dict[str, object]
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    transformed = {
        part: np.nan_to_num(values, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
        for part, values in transformed.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    artifacts['keep_mask'] = keep_mask
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, artifacts)
    return (transformed, artifacts)


def numerical_preprocess_v1(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v1 requires a seed in config.')
    train = x_num['train'].astype(np.float32, copy=False)
    medians = np.nanmedian(train, axis=0).astype(np.float32)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    imputed = {
        part: np.where(np.isnan(values), medians, values).astype(np.float32)
        for part, values in x_num.items()
    }
    low = np.nanpercentile(imputed['train'], 0.5, axis=0).astype(np.float32)
    high = np.nanpercentile(imputed['train'], 99.5, axis=0).astype(np.float32)
    clipped = {part: np.clip(values, low, high) for part, values in imputed.items()}
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(clipped['train'].shape[0] // 20, 1000), 10),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    transformer.fit(clipped['train'])
    transformed = {
        part: np.clip(transformer.transform(values), -5.0, 5.0)
        for part, values in clipped.items()
    }
    return _finish(
        transformed,
        {'transformer': transformer, 'medians': medians, 'low': low, 'high': high},
    )
