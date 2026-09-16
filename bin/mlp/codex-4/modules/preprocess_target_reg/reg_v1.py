# ruff: noqa
"""Standalone implementation for ``target_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np


def target_preprocess_v1(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Median/IQR target scaling for outlier-heavy regression."""
    median = float(np.median(y_raw['train']))
    q25, q75 = np.quantile(y_raw['train'], [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = float(y_raw['train'].std())
    if scale <= 1e-12:
        scale = 1.0
    y = {
        part: ((values - median) / scale).astype(np.float32)
        for part, values in y_raw.items()
    }

    def inverse_transform(values: np.ndarray) -> np.ndarray:
        limit = np.finfo(np.float32).max / 16.0
        restored = np.asarray(values, dtype=np.float64) * scale + median
        return np.clip(
            np.nan_to_num(restored, nan=median, posinf=limit, neginf=-limit),
            -limit,
            limit,
        )

    return (
        y,
        {
            'variant': 'robust_iqr',
            'median': median,
            'scale': scale,
            'inverse_transform': inverse_transform,
        },
    )
