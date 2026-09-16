"""Add small Gaussian noise to numeric training columns.

Adds `N(0, 0.05 * train_std)` noise to each numeric column on a copy of
the training data. Categorical columns are untouched. Labels are
preserved exactly so the augmentation is safe for both regression and
classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_NOISE_RATIO = 0.05


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def gaussian_noise_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = x_train.copy()
    num_cols = _numeric_cols(out)
    if not num_cols:
        return out, np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    for col in num_cols:
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(np.float32)
        finite = vals[np.isfinite(vals)]
        std = float(np.std(finite)) if finite.size else 0.0
        if std <= 0.0:
            continue
        noise = rng.normal(loc=0.0, scale=_NOISE_RATIO * std, size=vals.shape).astype(
            np.float32
        )
        out[col] = (vals + noise).astype(np.float32)
    return out, np.asarray(y_train).copy()
