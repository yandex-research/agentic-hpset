# ruff: noqa
"""Standalone implementation for ``target_preprocess_v1_minmax``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _minmax_inverse(values: np.ndarray, ymin: float, yrange: float) -> np.ndarray:
    return values * yrange + ymin


def target_preprocess_v1_minmax(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Min-max scale target to [-1, 1] using training-only statistics."""
    train = y_raw['train']
    ymin = float(train.min())
    ymax = float(train.max())
    yrange = ymax - ymin
    if yrange == 0.0:
        yrange = 1.0
    y = {
        part: ((values - ymin) / yrange * 2.0 - 1.0).astype(np.float32)
        for part, values in y_raw.items()
    }
    return (
        y,
        {
            'min': ymin,
            'max': ymax,
            'inverse_transform': lambda values,
            ymin=ymin,
            yrange=yrange: _minmax_inverse((values + 1.0) / 2.0, ymin, yrange),
        },
    )
