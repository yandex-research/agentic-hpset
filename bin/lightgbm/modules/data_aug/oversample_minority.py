"""Oversample minority classes to match the largest class.

If `y_train` looks like integer-valued classification labels with at most
64 unique values, the minority classes are oversampled with replacement so
that every class has the same count as the majority class. If the labels
don't look discrete, the input is returned unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MAX_CLASSES = 64


def _looks_classification(y: np.ndarray) -> bool:
    if y.size == 0:
        return False
    if not np.all(np.isfinite(y)):
        return False
    rounded = np.round(y)
    if not np.allclose(y, rounded, atol=1e-6):
        return False
    uniq = np.unique(rounded)
    if uniq.size > _MAX_CLASSES:
        return False
    if uniq.min() < 0:
        return False
    return True


def oversample_minority_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    y = np.asarray(y_train)
    if not _looks_classification(y):
        return x_train.copy(), y.copy()
    rng = np.random.default_rng(seed)
    rounded = np.round(y).astype(np.int64)
    classes, counts = np.unique(rounded, return_counts=True)
    target = int(counts.max())
    keep_idx: list[np.ndarray] = []
    for cls, count in zip(classes, counts):
        cls_idx = np.flatnonzero(rounded == cls)
        if count >= target:
            keep_idx.append(cls_idx)
            continue
        extra = rng.integers(0, cls_idx.size, size=target - count)
        keep_idx.append(np.concatenate([cls_idx, cls_idx[extra]]))
    final = np.concatenate(keep_idx)
    out_x = x_train.iloc[final].reset_index(drop=True)
    out_y = y[final].copy()
    return out_x, out_y
