# ruff: noqa
"""Standalone implementation for ``target_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _robust_inverse(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return values * scale + center


def target_preprocess_v1(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train = y_raw['train'].astype(np.float32, copy=False)
    center = float(np.median(train))
    q25, q75 = np.percentile(train, [25.0, 75.0])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-08:
        scale = float(train.std())
    if not np.isfinite(scale) or scale <= 1e-08:
        scale = 1.0
    y = {
        part: ((values - center) / scale).astype(np.float32)
        for part, values in y_raw.items()
    }
    return (
        y,
        {
            'center': center,
            'scale': scale,
            'inverse_transform': lambda values,
            center=center,
            scale=scale: _robust_inverse(values, center, scale),
        },
    )
