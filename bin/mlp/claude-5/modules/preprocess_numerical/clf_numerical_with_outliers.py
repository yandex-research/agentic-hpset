# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v2(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Quantile-transform features and concatenate per-feature outlier indicators.

    For each original feature, we add a binary indicator `|standardized| > 3.0`
    capturing tail/outlier values. The MLP can then route through these tails
    explicitly.
    """
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v2 requires a seed in config.')
    x_num_train = x_num['train']
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_num_train.shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    noisy_train = x_num_train + np.random.RandomState(seed).normal(
        0.0, 1e-05, x_num_train.shape
    ).astype(x_num_train.dtype)
    transformer.fit(noisy_train)
    median = np.median(x_num_train, axis=0)
    q1 = np.quantile(x_num_train, 0.25, axis=0)
    q3 = np.quantile(x_num_train, 0.75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-06)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        qt = np.nan_to_num(transformer.transform(values)).astype(np.float32)
        outlier = (np.abs(values - median) / iqr > 3.0).astype(np.float32)
        transformed[part] = np.concatenate([qt, outlier], axis=1)
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'transformer': transformer, 'keep_mask': keep_mask})
    return (transformed, {'transformer': transformer, 'keep_mask': keep_mask})
