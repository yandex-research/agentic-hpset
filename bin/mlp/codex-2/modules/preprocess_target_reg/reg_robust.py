# ruff: noqa
"""Standalone implementation for ``target_preprocess_robust``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _inverse_standard(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def target_preprocess_robust(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    median = float(np.median(y_raw['train']))
    q25, q75 = np.percentile(y_raw['train'], [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if scale <= 1e-12:
        scale = float(y_raw['train'].std()) or 1.0
    y = {
        part: ((values - median) / scale).astype(np.float32)
        for part, values in y_raw.items()
    }
    return (
        y,
        {
            'variant': 'robust',
            'median': median,
            'scale': scale,
            'inverse_transform': lambda values,
            median=median,
            scale=scale: _inverse_standard(values, median, scale),
        },
    )
