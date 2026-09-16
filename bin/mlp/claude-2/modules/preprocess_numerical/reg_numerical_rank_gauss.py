# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy.special import erfinv


def _rank_gauss_apply(
    values: np.ndarray, sorted_train: np.ndarray, gauss_train: np.ndarray
) -> np.ndarray:
    n_rows, n_cols = sorted_train.shape
    if n_rows == 0:
        return values.astype(np.float32)
    out = np.empty(values.shape, dtype=np.float32)
    for j in range(n_cols):
        col = values[:, j].astype(np.float64)
        ranks = np.searchsorted(sorted_train[:, j], col, side='left')
        ranks = np.clip(ranks, 0, n_rows - 1)
        out[:, j] = gauss_train[:, j][ranks].astype(np.float32)
    return out


def _rank_gauss_columns(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = train.shape
    if n_rows == 0:
        return (
            np.empty((0, n_cols), dtype=np.float64),
            np.empty((0, n_cols), dtype=np.float64),
        )
    sorted_train = np.sort(train, axis=0)
    eps = 1.0 / (4.0 * n_rows)
    cdf = (np.arange(1, n_rows + 1) - 0.5) / n_rows
    cdf = np.clip(cdf, eps, 1.0 - eps)
    gauss = np.sqrt(2.0) * erfinv(2.0 * cdf - 1.0)
    gauss_train = np.broadcast_to(gauss[:, None], (n_rows, n_cols)).copy()
    return (sorted_train, gauss_train)


def numerical_preprocess_v1(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'sorted_train': None, 'gauss_train': None, 'keep_mask': None})
    config = config or {}
    seed = config.get('seed', 0)
    train = x_num['train'].astype(np.float64)
    rng = np.random.RandomState(int(seed))
    train_noisy = train + rng.normal(0.0, 1e-05, train.shape)
    sorted_train, gauss_train = _rank_gauss_columns(train_noisy)
    transformed = {
        part: _rank_gauss_apply(values, sorted_train, gauss_train)
        for part, values in x_num.items()
    }
    transformed = {
        part: np.nan_to_num(values).astype(np.float32)
        for part, values in transformed.items()
    }
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {
                'sorted_train': sorted_train,
                'gauss_train': gauss_train,
                'keep_mask': None,
            },
        )
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {
                'sorted_train': sorted_train,
                'gauss_train': gauss_train,
                'keep_mask': keep_mask,
            },
        )
    return (
        transformed,
        {
            'sorted_train': sorted_train,
            'gauss_train': gauss_train,
            'keep_mask': keep_mask,
        },
    )
