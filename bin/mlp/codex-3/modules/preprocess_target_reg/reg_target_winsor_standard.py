# ruff: noqa
"""Standalone implementation for ``target_preprocess_v4``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _inverse(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def target_preprocess_v4(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train = y_raw['train'].astype(np.float32, copy=False)
    low, high = np.percentile(train, [0.5, 99.5])
    clipped_train = np.clip(train, low, high)
    mean = float(clipped_train.mean())
    std = float(clipped_train.std())
    if not np.isfinite(std) or std <= 1e-08:
        std = 1.0
    y = {
        part: ((np.clip(values, low, high) - mean) / std).astype(np.float32)
        for part, values in y_raw.items()
    }
    return (
        y,
        {
            'low': float(low),
            'high': float(high),
            'mean': mean,
            'std': std,
            'inverse_transform': lambda values, mean=mean, std=std: _inverse(
                values, mean, std
            ),
        },
    )
