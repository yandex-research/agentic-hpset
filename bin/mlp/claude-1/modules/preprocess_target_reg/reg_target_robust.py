# ruff: noqa
"""Standalone implementation for ``target_preprocess_robust``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _robust_inverse(values: np.ndarray, median: float, mad: float) -> np.ndarray:
    return values * mad + median


def target_preprocess_robust(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train = y_raw['train']
    median = float(np.median(train))
    mad = float(np.median(np.abs(train - median))) * 1.4826
    if mad == 0.0:
        mad = float(train.std()) or 1.0
    y = {
        part: ((values - median) / mad).astype(np.float32)
        for part, values in y_raw.items()
    }
    return (
        y,
        {
            'median': median,
            'mad': mad,
            'inverse_transform': lambda values, median=median, mad=mad: _robust_inverse(
                values, median, mad
            ),
        },
    )
