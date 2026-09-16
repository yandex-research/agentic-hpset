# ruff: noqa
"""Standalone implementation for ``target_preprocess_log``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _log_inverse(
    values: np.ndarray, mean: float, std: float, shift: float
) -> np.ndarray:
    raw = values * std + mean
    return np.expm1(raw) - shift


def target_preprocess_log(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_min = float(y_raw['train'].min())
    shift = max(0.0, -train_min) + 1e-06
    log_y = {
        part: np.log1p(values + shift).astype(np.float64)
        for part, values in y_raw.items()
    }
    mean = float(log_y['train'].mean())
    std = float(log_y['train'].std())
    if std == 0.0:
        std = 1.0
    y = {
        part: ((values - mean) / std).astype(np.float32)
        for part, values in log_y.items()
    }
    return (
        y,
        {
            'mean': mean,
            'std': std,
            'shift': shift,
            'inverse_transform': lambda values,
            mean=mean,
            std=std,
            shift=shift: _log_inverse(values, mean, std, shift),
        },
    )
