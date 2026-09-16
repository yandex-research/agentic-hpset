from __future__ import annotations

from typing import Any

import numpy as np

from bin.tabm.modules._shared import column_centers, fill_nans, filter_constant_columns


def numerical_preprocess_v3(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Robust IQR scaling with hard clipping at +/- 5.

    Hypothesis: a handful of extreme rows in raw tabular features can dominate
    the optimizer's first updates. Centering on median and dividing by IQR
    bounds the influence of outliers without the harsh rank-collapse of quantile
    transforms. Clipping at +/- 5 sigma-equivalents prevents pathological
    activations while leaving room for the model to distinguish neighbors.
    """
    if x_num is None:
        return None, {"median": None, "iqr": None, "keep_mask": None}
    train = np.asarray(x_num["train"], dtype=np.float64)
    median = column_centers(train, np.nanmedian)
    train_filled = fill_nans(train, median)
    q25 = np.quantile(train_filled, 0.25, axis=0)
    q75 = np.quantile(train_filled, 0.75, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr < 1e-8, 1.0, iqr)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        arr = fill_nans(np.asarray(values, dtype=np.float64), median)
        scaled = np.clip((arr - median) / iqr, -5.0, 5.0)
        transformed[part] = scaled.astype(np.float32)
    transformed, keep_mask = filter_constant_columns(transformed)
    artifacts = {"median": median, "iqr": iqr, "keep_mask": keep_mask}
    if transformed["train"].shape[1] == 0:
        return None, artifacts
    return transformed, artifacts
