# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v5_rank_gauss``."""

from __future__ import annotations
from typing import Any
from scipy.special import erfinv
import numpy as np


def _rank_gauss_transform(values: np.ndarray, sorted_train: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    n_train = sorted_train.shape[0]
    for col in range(values.shape[1]):
        ranks = np.searchsorted(sorted_train[:, col], values[:, col], side='left')
        ranks = np.clip(ranks, 0, n_train - 1)
        normalized = (ranks.astype(np.float64) + 0.5) / n_train
        normalized = np.clip(normalized, 1e-06, 1.0 - 1e-06)
        out[:, col] = np.sqrt(2.0) * erfinv(2.0 * normalized - 1.0)
    return out


def numerical_preprocess_v5_rank_gauss(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'sorted_train': None, 'keep_mask': None})
    sorted_train = np.sort(x_num['train'], axis=0)
    transformed = {
        part: _rank_gauss_transform(values, sorted_train).astype(np.float32)
        for part, values in x_num.items()
    }
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
    return (transformed, {'sorted_train': sorted_train, 'keep_mask': keep_mask})
