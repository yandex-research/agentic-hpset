# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v5``."""

from __future__ import annotations
from typing import Any
from scipy.special import erfinv
import numpy as np


def _rank_gauss_apply(
    values: np.ndarray, sorted_uniques: list[np.ndarray]
) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    for col_idx, column in enumerate(values.T):
        anchor = sorted_uniques[col_idx]
        n = anchor.shape[0]
        if n <= 1:
            out[:, col_idx] = 0.0
            continue
        ranks = np.searchsorted(anchor, column, side='left').astype(np.float64)
        ranks = np.clip(ranks, 0, n - 1)
        normalized = ranks / max(n - 1, 1) * 2.0 - 1.0
        normalized = np.clip(normalized, -0.999999, 0.999999)
        out[:, col_idx] = (np.sqrt(2.0) * erfinv(normalized)).astype(np.float32)
    return out


def _rank_gauss_fit(x_train: np.ndarray) -> list[np.ndarray]:
    return [np.sort(np.unique(column)) for column in x_train.T]


def numerical_preprocess_v5(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'sorted_uniques': None, 'keep_mask': None})
    sorted_uniques = _rank_gauss_fit(x_num['train'])
    transformed = {
        part: np.nan_to_num(_rank_gauss_apply(values, sorted_uniques))
        for part, values in x_num.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'sorted_uniques': sorted_uniques, 'keep_mask': keep_mask})
    return (transformed, {'sorted_uniques': sorted_uniques, 'keep_mask': keep_mask})
