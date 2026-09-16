"""Duplicate a random subset of training rows.

Picks 20% of training rows uniformly at random and appends a copy of
those rows (with their labels) to the training set. Cheap, label-safe,
and reproducible per seed. Acts as a light bias-shift toward the
empirical distribution without modifying any feature values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DUP_RATE = 0.20


def random_duplicate_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(x_train)
    y = np.asarray(y_train)
    if n == 0:
        return x_train.copy(), y.copy()
    rng = np.random.default_rng(seed)
    k = int(round(_DUP_RATE * n))
    if k <= 0:
        return x_train.copy(), y.copy()
    dup_idx = rng.integers(0, n, size=k)
    out_x = pd.concat(
        [x_train, x_train.iloc[dup_idx]], axis=0, ignore_index=True
    )
    out_y = np.concatenate([y, y[dup_idx]]).copy()
    return out_x, out_y
