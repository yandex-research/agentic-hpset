from __future__ import annotations

from typing import Any

import numpy as np

from bin.tabm.modules._shared import signed_expm1, signed_log1p


def _inverse_transform(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    log_space = values * std + mean
    return signed_expm1(log_space).astype(np.float32)


def target_preprocess_v1(
    y_raw: dict[str, np.ndarray],
    _config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Signed log1p transform on the target, then standardize.

    Hypothesis: many regression targets (housing prices, durations, counts) are
    right-skewed; learning in log-space stabilizes the loss surface and reduces the
    effect of large outliers, while standardizing keeps gradients well-scaled.
    """
    log_train = signed_log1p(y_raw["train"])
    mean = float(log_train.mean())
    std = float(log_train.std())
    if std == 0.0:
        std = 1.0
    y = {
        part: ((signed_log1p(values) - mean) / std).astype(np.float32)
        for part, values in y_raw.items()
    }
    return y, {
        "mean": mean,
        "std": std,
        "inverse_transform": lambda values, mean=mean, std=std: _inverse_transform(
            values, mean, std
        ),
    }
