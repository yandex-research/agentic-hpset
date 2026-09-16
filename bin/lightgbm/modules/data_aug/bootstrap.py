"""Plain bootstrap resampling of the training rows.

Samples `n_train` rows from `(x_train, y_train)` with replacement. The
ensemble-style row diversity nudges LightGBM toward fitting a slightly
different empirical distribution per seed; together with the model's
existing `bagging_fraction` this acts as a complementary stochastic
regularizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(x_train)
    if n == 0:
        return x_train.copy(), np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=n)
    out_x = x_train.iloc[idx].reset_index(drop=True)
    out_y = np.asarray(y_train)[idx].copy()
    return out_x, out_y
