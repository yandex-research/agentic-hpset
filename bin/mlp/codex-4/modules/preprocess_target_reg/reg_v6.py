# ruff: noqa
"""Standalone implementation for ``target_preprocess_v6``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def target_preprocess_v6(
    y_raw: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Yeo-Johnson target power transform for skewed regression labels."""
    transformer = sklearn.preprocessing.PowerTransformer(
        method='yeo-johnson', standardize=True
    ).fit(y_raw['train'].reshape(-1, 1))
    y = {
        part: transformer.transform(values.reshape(-1, 1)).ravel().astype(np.float32)
        for part, values in y_raw.items()
    }
    train_transformed = np.asarray(y['train'], dtype=np.float64)
    inverse_lower = float(train_transformed.min())
    inverse_upper = float(train_transformed.max())

    def inverse_transform(values: np.ndarray) -> np.ndarray:
        bounded = np.nan_to_num(
            np.asarray(values, dtype=np.float64),
            nan=0.0,
            posinf=inverse_upper,
            neginf=inverse_lower,
        )
        bounded = np.clip(bounded, inverse_lower, inverse_upper)
        restored = transformer.inverse_transform(bounded.reshape(-1, 1)).ravel()
        limit = np.finfo(np.float32).max / 16.0
        fallback = float(np.median(y_raw['train']))
        return np.clip(
            np.nan_to_num(restored, nan=fallback, posinf=limit, neginf=-limit),
            -limit,
            limit,
        )

    return (
        y,
        {
            'variant': 'yeo_johnson',
            'transformer': transformer,
            'inverse_lower': inverse_lower,
            'inverse_upper': inverse_upper,
            'inverse_transform': inverse_transform,
        },
    )
