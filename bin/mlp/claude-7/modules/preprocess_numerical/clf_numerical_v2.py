# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v2(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """RankGauss-style transform plus per-feature bin-index numerical augmentation.

    For each numerical column, we (1) apply a quantile-to-Gaussian transform
    (RankGauss-equivalent via QuantileTransformer with 'normal') and (2) append
    the integer bin index normalized to [0, 1] as an additional numerical
    column. The bin index injects a coarse, monotone discretized signal that
    the downstream PLR embedding can exploit alongside the smooth rank-Gaussian
    encoding.
    """
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None, 'bin_edges': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v2 requires a seed in config.')
    x_num_train = x_num['train']
    n_quantiles = max(min(x_num_train.shape[0] // 30, 1000), 10)
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    noisy_train = x_num_train + np.random.RandomState(seed).normal(
        0.0, 1e-05, x_num_train.shape
    ).astype(x_num_train.dtype)
    transformer.fit(noisy_train)
    transformed = {
        part: transformer.transform(values) for part, values in x_num.items()
    }
    transformed = {
        part: np.nan_to_num(values).astype(np.float32)
        for part, values in transformed.items()
    }
    n_bins = 16
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(x_num_train, quantiles, axis=0).astype(np.float32)
    bin_features: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        binned = np.empty_like(values, dtype=np.float32)
        for col in range(values.shape[1]):
            edges = np.unique(bin_edges[:, col])
            if edges.size <= 1:
                binned[:, col] = 0.0
            else:
                idx = np.clip(
                    np.searchsorted(edges[1:-1], values[:, col]), 0, edges.size - 2
                )
                binned[:, col] = idx.astype(np.float32) / max(edges.size - 2, 1)
        bin_features[part] = binned
    augmented = {
        part: np.concatenate([transformed[part], bin_features[part]], axis=1).astype(
            np.float32
        )
        for part in transformed
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in augmented['train'].T], dtype=bool
    )
    augmented = {part: values[:, keep_mask] for part, values in augmented.items()}
    if augmented['train'].shape[1] == 0:
        return (
            None,
            {
                'transformer': transformer,
                'keep_mask': keep_mask,
                'bin_edges': bin_edges,
            },
        )
    return (
        augmented,
        {'transformer': transformer, 'keep_mask': keep_mask, 'bin_edges': bin_edges},
    )
