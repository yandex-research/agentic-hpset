# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v6``."""

from __future__ import annotations
from typing import Any
import numpy as np
from ...core import bounded_numerical_parts


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if len(parts) == 1:
        return {part: values.astype(np.float32) for part, values in parts[0].items()}
    return {
        part: np.concatenate([block[part] for block in parts], axis=1).astype(
            np.float32
        )
        for part in parts[0]
    }


def _drop_constant(
    transformed: dict[str, np.ndarray], artifacts: dict[str, object]
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    train = np.nan_to_num(transformed['train'], nan=0.0, posinf=0.0, neginf=0.0)
    keep_mask = train.max(axis=0) - train.min(axis=0) > 1e-12
    cleaned = {
        part: np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)[
            :, keep_mask
        ]
        for part, values in transformed.items()
    }
    artifacts = {**artifacts, 'keep_mask': keep_mask}
    if cleaned['train'].shape[1] == 0:
        return (None, artifacts)
    return (cleaned, artifacts)


def _median_impute(
    x_num: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    train = x_num['train'].astype(np.float32, copy=False)
    median = np.nanmedian(train, axis=0).astype(np.float32)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    return (
        {
            part: np.where(np.isnan(values), median, values).astype(np.float32)
            for part, values in x_num.items()
        },
        median,
    )


def _robust_stats(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q25, q75 = np.quantile(train, [0.25, 0.75], axis=0)
    median = np.median(train, axis=0)
    iqr = q75 - q25
    std = train.std(axis=0)
    scale = np.where(iqr > 1e-06, iqr, std)
    scale = np.where(scale > 1e-06, scale, 1.0).astype(np.float32)
    return (median.astype(np.float32), scale)


def numerical_preprocess_v6(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Signed-log robust scaling concatenated with robust raw scaling."""
    if x_num is None:
        return (None, {'variant': 'signed_log_robust', 'keep_mask': None})
    bounded, support = bounded_numerical_parts(x_num)
    center, scale = _robust_stats(bounded['train'])
    robust = {part: (values - center) / scale for part, values in bounded.items()}
    signed_log = {
        part: np.sign(values) * np.log1p(np.abs(values))
        for part, values in robust.items()
    }
    return _drop_constant(
        _concat([signed_log, robust]),
        {
            'variant': 'signed_log_robust',
            'median': support['fill'],
            'center': center,
            'scale': scale,
            **support,
        },
    )
