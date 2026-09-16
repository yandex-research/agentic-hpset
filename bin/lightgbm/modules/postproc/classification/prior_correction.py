"""Bayes-style prior correction toward a flat class prior.

Divides each class probability by the train marginal `P_train(class)`
and renormalizes the row. The corrected probabilities are what the
model "would have" output if every class was equally likely in train.
Useful when the deployment distribution is closer to uniform than the
imbalanced training set. Uses only train labels, so no val overfit.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _binarize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.column_stack([1.0 - arr, arr])
    return arr


def _train_prior(y_train: np.ndarray, k: int) -> np.ndarray:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    counts = np.zeros(k, dtype=np.float64)
    rounded = np.round(y).astype(np.int64)
    valid = (rounded >= 0) & (rounded < k)
    for c in rounded[valid]:
        counts[int(c)] += 1.0
    total = counts.sum()
    if total <= 0:
        return np.full(k, 1.0 / k)
    return np.maximum(counts / total, _EPS)


def prior_correction_postproc(
    proba: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    first = next(iter(proba.values()))
    k = _binarize(np.asarray(first)).shape[1]
    prior = _train_prior(y_train, k)
    for part, arr in proba.items():
        a = _binarize(np.asarray(arr, dtype=np.float64))
        adjusted = a / prior[None, :]
        row_sum = adjusted.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        out[part] = (adjusted / row_sum).astype(np.float32)
    return out
