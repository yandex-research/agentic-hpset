from __future__ import annotations

from typing import Any

import numpy as np


def _log_inverse(values: np.ndarray, mean: float, std: float, offset: float) -> np.ndarray:
    return np.exp(values * std + mean) + offset


def target_preprocess_v4(
    y_raw: dict[str, np.ndarray],
    _config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Train-only log target scaling with standardization.

    Picks an additive offset so `log(y - offset)` is well-defined for all
    train labels (`offset = min(train_min, 0) - 1`), takes the log on shifted
    targets, then z-scores against train stats. Best for positive,
    right-skewed regression targets (times, durations, prices). RMSE/MAE in
    the raw target scale are recovered via `inverse_transform`.
    """
    train_min = float(y_raw["train"].min())
    offset = min(train_min, 0.0) - 1.0
    y_log_train = np.log(np.maximum(y_raw["train"] - offset, 1e-6))
    mean = float(y_log_train.mean())
    std = float(y_log_train.std())
    if std == 0.0:
        std = 1.0
    y = {
        part: ((np.log(np.maximum(values - offset, 1e-6)) - mean) / std).astype(np.float32)
        for part, values in y_raw.items()
    }
    return y, {
        "offset": offset,
        "mean": mean,
        "std": std,
        "inverse_transform": lambda values, mean=mean, std=std, offset=offset: _log_inverse(
            values, mean, std, offset
        ),
    }
