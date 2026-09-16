# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v3(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Concatenate quantile-transformed features with rank-uniform features.

    Adds a uniform-distribution view of each feature to complement the normal-
    distribution view from QuantileTransformer; this gives the model both
    bounded and unbounded representations of the same signal.
    """
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v3 requires a seed in config.')
    x_num_train = x_num['train']
    n_quantiles = max(min(x_num_train.shape[0] // 30, 1000), 10)
    normal_transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    uniform_transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution='uniform',
        subsample=1000000000,
        random_state=seed,
    )
    noisy_train = x_num_train + np.random.RandomState(seed).normal(
        0.0, 1e-05, x_num_train.shape
    ).astype(x_num_train.dtype)
    normal_transformer.fit(noisy_train)
    uniform_transformer.fit(noisy_train)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        normal = np.nan_to_num(normal_transformer.transform(values)).astype(np.float32)
        uniform = np.nan_to_num(uniform_transformer.transform(values)).astype(
            np.float32
        )
        transformed[part] = np.concatenate([normal, uniform], axis=1)
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {
                'transformer': normal_transformer,
                'uniform_transformer': uniform_transformer,
                'keep_mask': keep_mask,
            },
        )
    return (
        transformed,
        {
            'transformer': normal_transformer,
            'uniform_transformer': uniform_transformer,
            'keep_mask': keep_mask,
        },
    )
