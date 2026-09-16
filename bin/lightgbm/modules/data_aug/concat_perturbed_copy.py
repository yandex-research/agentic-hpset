"""Concatenate the original training data with a noise-perturbed copy.

Returns `2 * n_train` rows: the unchanged training data followed by a
copy where each numeric column has independent Gaussian noise of
magnitude `0.05 * train_std`. Labels are duplicated. Doubles the
training-data memory footprint, so prefer cheaper modules unless this
broader regularization is needed.
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


def concat_perturbed_copy_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(x_train)
    y = np.asarray(y_train)
    if n == 0:
        return x_train.copy(), y.copy()
    rng = np.random.default_rng(seed)
    perturbed = x_train.copy()
    for col in _numeric_cols(perturbed):
        vals = pd.to_numeric(perturbed[col], errors="coerce").to_numpy(np.float32)
        finite = vals[np.isfinite(vals)]
        std = float(np.std(finite)) if finite.size else 0.0
        if std <= 0.0:
            continue
        noise = rng.normal(0.0, _NOISE_RATIO * std, size=vals.shape).astype(np.float32)
        perturbed[col] = (vals + noise).astype(np.float32)
    out_x = pd.concat([x_train, perturbed], axis=0, ignore_index=True)
    out_y = np.concatenate([y, y]).copy()
    return out_x, out_y
