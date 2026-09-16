"""Blend predicted probabilities with the train class prior.

Returns `0.95 * proba + 0.05 * train_prior`. The small mix toward the
empirical training prior is a stable shrinkage anchor — it nudges
overconfident probabilities back toward the marginal without ever
fitting validation. Compared with prior correction, this *pulls toward*
the prior rather than *correcting it out*.
"""

from __future__ import annotations

import numpy as np

_BLEND = 0.05
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


def train_prior_blend_postproc(
    proba: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    first = next(iter(proba.values()))
    k = _binarize(np.asarray(first)).shape[1]
    prior = _train_prior(y_train, k)
    out: dict[str, np.ndarray] = {}
    for part, arr in proba.items():
        a = _binarize(np.asarray(arr, dtype=np.float64))
        blended = (1.0 - _BLEND) * a + _BLEND * prior[None, :]
        row_sum = blended.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        out[part] = (blended / row_sum).astype(np.float32)
    return out
