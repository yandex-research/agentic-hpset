from __future__ import annotations

from typing import Any

import numpy as np


def _standardize_inverse(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def target_preprocess_v0(
    y_raw: dict[str, np.ndarray],
    _config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    mean = float(y_raw["train"].mean())
    std = float(y_raw["train"].std())
    if std == 0.0:
        std = 1.0
    y = {
        part: ((values - mean) / std).astype(np.float32)
        for part, values in y_raw.items()
    }
    return y, {
        "mean": mean,
        "std": std,
        "inverse_transform": lambda values, mean=mean, std=std: _standardize_inverse(
            values, mean, std
        ),
    }
