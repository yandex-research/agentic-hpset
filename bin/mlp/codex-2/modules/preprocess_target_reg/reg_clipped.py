# ruff: noqa
"""Standalone implementation for ``target_preprocess_clipped``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _inverse_standard(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def _standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def target_preprocess_clipped(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    lower, upper = np.percentile(y_raw['train'], [0.5, 99.5])
    clipped = {
        part: np.clip(values, lower, upper).astype(np.float32)
        for part, values in y_raw.items()
    }
    mean = float(clipped['train'].mean())
    std = float(clipped['train'].std())
    if std == 0.0:
        std = 1.0
    y = {part: _standardize(values, mean, std) for part, values in clipped.items()}
    return (
        y,
        {
            'variant': 'clipped',
            'lower': float(lower),
            'upper': float(upper),
            'mean': mean,
            'std': std,
            'inverse_transform': lambda values, mean=mean, std=std: _inverse_standard(
                values, mean, std
            ),
        },
    )
