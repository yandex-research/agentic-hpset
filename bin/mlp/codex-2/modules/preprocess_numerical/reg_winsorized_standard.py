# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_winsorized_standard``."""

from __future__ import annotations
from typing import Any
import numpy as np


def _as_float32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _drop_constant(
    transformed: dict[str, np.ndarray], *, artifact: dict[str, object]
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    train = transformed['train']
    keep_mask = np.array([np.nanstd(column) > 1e-07 for column in train.T], dtype=bool)
    artifact['keep_mask'] = keep_mask
    if keep_mask.size == 0 or not keep_mask.any():
        return (None, artifact)
    return (
        {
            part: values[:, keep_mask].astype(np.float32)
            for part, values in transformed.items()
        },
        artifact,
    )


def _empty_artifacts(name: str) -> dict[str, object]:
    return {'variant': name, 'keep_mask': None}


def _impute_with_medians(
    x_num: dict[str, np.ndarray], medians: np.ndarray
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        values = _as_float32(values).copy()
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(medians, np.where(missing)[1])
        result[part] = values
    return result


def _standardize_from_train(
    values: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mean = values['train'].mean(axis=0).astype(np.float32)
    std = values['train'].std(axis=0).astype(np.float32)
    std = np.where(std > 1e-06, std, 1.0).astype(np.float32)
    return (
        {
            part: np.nan_to_num((part_values - mean) / std).astype(np.float32)
            for part, part_values in values.items()
        },
        mean,
        std,
    )


def _train_medians(x_train: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(x_train, axis=0).astype(np.float32)
    return np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)


def numerical_preprocess_winsorized_standard(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, _empty_artifacts('winsorized_standard'))
    medians = _train_medians(_as_float32(x_num['train']))
    imputed = _impute_with_medians(x_num, medians)
    lower = np.percentile(imputed['train'], 1.0, axis=0).astype(np.float32)
    upper = np.percentile(imputed['train'], 99.0, axis=0).astype(np.float32)
    clipped = {
        part: np.clip(values, lower, upper).astype(np.float32)
        for part, values in imputed.items()
    }
    standardized, mean, std = _standardize_from_train(clipped)
    return _drop_constant(
        standardized,
        artifact={
            'variant': 'winsorized_standard',
            'medians': medians,
            'lower': lower,
            'upper': upper,
            'mean': mean,
            'std': std,
        },
    )
