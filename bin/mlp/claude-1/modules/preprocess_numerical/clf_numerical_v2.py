# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy.special import erfinv


def _rank_gauss_fit(x_train: np.ndarray) -> dict[str, np.ndarray]:
    n = x_train.shape[0]
    sorted_train = np.sort(x_train, axis=0)
    return {'sorted_train': sorted_train, 'n_train': np.array([n], dtype=np.int64)}


def _rank_gauss_transform(x: np.ndarray, sorted_train: np.ndarray) -> np.ndarray:
    n_train = sorted_train.shape[0]
    n_features = x.shape[1]
    out = np.empty_like(x, dtype=np.float64)
    for j in range(n_features):
        ranks = np.searchsorted(sorted_train[:, j], x[:, j], side='left')
        ranks = np.clip(ranks, 0, n_train)
        u = (ranks + 0.5) / (n_train + 1.0)
        u = np.clip(u, 1e-06, 1.0 - 1e-06)
        out[:, j] = np.sqrt(2.0) * erfinv(2.0 * u - 1.0)
    return out


def numerical_preprocess_v2(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'sorted_train': None, 'keep_mask': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v2 requires a seed in config.')
    x_train = x_num['train'].astype(np.float64)
    rng = np.random.RandomState(seed)
    low = np.quantile(x_train, 0.01, axis=0)
    high = np.quantile(x_train, 0.99, axis=0)
    eq = high <= low
    if eq.any():
        col_min = x_train.min(axis=0)
        col_max = x_train.max(axis=0)
        low = np.where(eq, col_min, low)
        high = np.where(eq, col_max, high)
    x_train_clipped = np.clip(x_train, low, high)
    x_train_clipped = x_train_clipped + rng.normal(0.0, 1e-05, x_train_clipped.shape)
    artifacts = _rank_gauss_fit(x_train_clipped)
    sorted_train = artifacts['sorted_train']
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        clipped = np.clip(values.astype(np.float64), low, high)
        transformed[part] = _rank_gauss_transform(clipped, sorted_train)
    transformed = {
        part: np.nan_to_num(values).astype(np.float32)
        for part, values in transformed.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'sorted_train': sorted_train, 'keep_mask': keep_mask})
    return (
        transformed,
        {
            'sorted_train': sorted_train,
            'keep_mask': keep_mask,
            'low': low,
            'high': high,
        },
    )
