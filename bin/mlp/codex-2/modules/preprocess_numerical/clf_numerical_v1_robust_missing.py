# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _column_stats(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lows: list[float] = []
    highs: list[float] = []
    centers: list[float] = []
    scales: list[float] = []
    for column in values.T:
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            lows.append(-1.0)
            highs.append(1.0)
            centers.append(0.0)
            scales.append(1.0)
            continue
        low, high = np.quantile(finite, [0.005, 0.995])
        center = np.median(finite)
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        scale = q75 - q25
        if not np.isfinite(scale) or scale < 1e-06:
            scale = np.std(finite)
        if not np.isfinite(scale) or scale < 1e-06:
            scale = 1.0
        lows.append(float(low))
        highs.append(float(high))
        centers.append(float(center))
        scales.append(float(scale))
    return (
        np.asarray(lows, dtype=np.float32),
        np.asarray(highs, dtype=np.float32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


def numerical_preprocess_v1(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (
            None,
            {
                'low': None,
                'high': None,
                'center': None,
                'scale': None,
                'keep_mask': None,
            },
        )
    train = np.asarray(x_num['train'], dtype=np.float32)
    low, high, center, scale = _column_stats(train)
    missing_columns = np.isnan(train).any(axis=0)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        values = np.asarray(values, dtype=np.float32)
        missing = np.isnan(values).astype(np.float32)
        imputed = np.where(np.isfinite(values), values, center)
        clipped = np.clip(imputed, low, high)
        scaled = np.clip((clipped - center) / scale, -8.0, 8.0).astype(np.float32)
        pieces = [scaled]
        if missing_columns.any():
            pieces.append(missing[:, missing_columns])
        transformed[part] = np.column_stack(pieces).astype(np.float32)
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {
                'low': low,
                'high': high,
                'center': center,
                'scale': scale,
                'keep_mask': keep_mask,
            },
        )
    return (
        transformed,
        {
            'low': low,
            'high': high,
            'center': center,
            'scale': scale,
            'keep_mask': keep_mask,
        },
    )
