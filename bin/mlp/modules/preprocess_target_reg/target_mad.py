from __future__ import annotations

from typing import Any

import numpy as np


def _mad_inverse(values: np.ndarray, median: float, mad: float) -> np.ndarray:
    return values * mad + median


def target_preprocess_v3(
    y_raw: dict[str, np.ndarray],
    _config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Robust target scaling using median and MAD (Median Absolute Deviation).

    Standardizes as (y - median) / MAD instead of (y - mean) / std. The
    breakdown point is 50% (vs 0% for mean/std), meaning up to half the
    data can be outliers without distorting the scale estimate. Especially
    valuable for targets with heavy tails or contaminated distributions
    where mean/std-based standardization produces poor conditioning.
    """
    median = float(np.median(y_raw["train"]))
    mad = float(np.median(np.abs(y_raw["train"] - median)))
    if mad == 0.0:
        # Fallback to std if MAD is zero (e.g., constant-heavy distributions)
        mad = float(y_raw["train"].std())
    if mad == 0.0:
        mad = 1.0
    # Scale MAD to be consistent with std for normal distributions
    # MAD * 1.4826 ≈ std for Gaussian
    mad_scaled = mad * 1.4826
    y = {
        part: ((values - median) / mad_scaled).astype(np.float32)
        for part, values in y_raw.items()
    }
    return y, {
        "median": median,
        "mad_scaled": mad_scaled,
        "inverse_transform": lambda values, median=median, mad_scaled=mad_scaled: _mad_inverse(
            values, median, mad_scaled
        ),
    }
