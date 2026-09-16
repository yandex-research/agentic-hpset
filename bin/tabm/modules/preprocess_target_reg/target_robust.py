from __future__ import annotations

from typing import Any

import numpy as np


def _inverse_transform(values: np.ndarray, median: float, scale: float) -> np.ndarray:
    return (values * scale + median).astype(np.float32)


def target_preprocess_v3(
    y_raw: dict[str, np.ndarray],
    _config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Center y on the training median and rescale by 1.4826 * MAD.

    Hypothesis: extreme labels dominate MSE training under classic standardization,
    pulling the network toward the tails. Median + MAD is robust to a few outliers
    so the model focuses on the bulk while the inverse transform recovers the
    original scale faithfully.
    """
    train = np.asarray(y_raw["train"], dtype=np.float64)
    median = float(np.median(train))
    mad = float(np.median(np.abs(train - median)))
    scale = 1.4826 * mad
    if scale == 0.0:
        scale = float(train.std())
        if scale == 0.0:
            scale = 1.0
    y = {
        part: ((np.asarray(values, dtype=np.float64) - median) / scale).astype(np.float32)
        for part, values in y_raw.items()
    }
    return y, {
        "median": median,
        "scale": scale,
        "inverse_transform": lambda values, median=median, scale=scale: _inverse_transform(
            values, median, scale
        ),
    }
