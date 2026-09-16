from __future__ import annotations

from typing import Any

import numpy as np


def _symlog(x: np.ndarray) -> np.ndarray:
    """Symmetric log: sign(x) * log(1 + |x|). Handles positive and negative values."""
    return np.sign(x) * np.log1p(np.abs(x))


def _symlog_inv(x: np.ndarray) -> np.ndarray:
    """Inverse of symmetric log: sign(x) * (exp(|x|) - 1)."""
    return np.sign(x) * np.expm1(np.abs(x))


def target_preprocess_v2(
    y_raw: dict[str, np.ndarray],
    _config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Symmetric log transform + standardization for targets.

    Applies sign(y)*log(1+|y|) before standardizing. This compresses large
    values while preserving sign and zero. Especially effective for targets
    with heavy right tails (prices, counts, incomes) where standardization
    alone still leaves a skewed loss landscape.

    The log transform makes the loss more uniform across scales: errors on
    large targets are penalized less harshly in absolute terms, but more
    harshly in relative terms.
    """
    # Apply symlog first
    y_log = {part: _symlog(values) for part, values in y_raw.items()}
    # Then standardize
    mean = float(y_log["train"].mean())
    std = float(y_log["train"].std())
    if std == 0.0:
        std = 1.0
    y = {
        part: ((values - mean) / std).astype(np.float32)
        for part, values in y_log.items()
    }

    def inverse_transform(values: np.ndarray, mean=mean, std=std) -> np.ndarray:
        unstandardized = values * std + mean
        return _symlog_inv(unstandardized)

    return y, {
        "mean": mean,
        "std": std,
        "inverse_transform": inverse_transform,
    }
