"""Add small uniform jitter to numeric training columns.

Adds uniform `U(-0.05 * std, 0.05 * std)` noise to each numeric column.
Heavier in the tails than Gaussian for a given std and bounded, so it
keeps the perturbed values in a tight neighborhood of the original even
on long-tailed features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_JITTER_RATIO = 0.05


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def uniform_jitter_aug(
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
        scale = _JITTER_RATIO * std
        jitter = rng.uniform(-scale, scale, size=vals.shape).astype(np.float32)
        out[col] = (vals + jitter).astype(np.float32)
    return out, np.asarray(y_train).copy()
