# ruff: noqa
"""Standalone implementation for ``target_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np


def target_preprocess_v2(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Winsorized standard target for skewed, noisy regression labels."""
    lo, hi = np.quantile(y_raw['train'], [0.01, 0.99])
    clipped_train = np.clip(y_raw['train'], lo, hi)
    mean = float(clipped_train.mean())
    std = float(clipped_train.std())
    if std <= 1e-12:
        std = 1.0
    y = {
        part: ((np.clip(values, lo, hi) - mean) / std).astype(np.float32)
        for part, values in y_raw.items()
    }

    def inverse_transform(values: np.ndarray) -> np.ndarray:
        limit = np.finfo(np.float32).max / 16.0
        restored = np.asarray(values, dtype=np.float64) * std + mean
        return np.clip(
            np.nan_to_num(restored, nan=mean, posinf=limit, neginf=-limit),
            -limit,
            limit,
        )

    return (
        y,
        {
            'variant': 'winsor_standard',
            'lo': float(lo),
            'hi': float(hi),
            'mean': mean,
            'std': std,
            'inverse_transform': inverse_transform,
        },
    )
