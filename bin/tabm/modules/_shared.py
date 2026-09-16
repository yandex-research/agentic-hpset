"""Utilities shared across preprocessing modules.

Kept deliberately small: only abstractions that were copy-pasted in three or
more places, with no module-specific logic baked in.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import sklearn.preprocessing


_QUANTILE_JITTER_STD = 1e-5


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def signed_expm1(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.expm1(np.abs(values))


def add_quantile_jitter(x_train: np.ndarray, seed: int) -> np.ndarray:
    """Add tiny Gaussian noise so quantile-based fits don't choke on ties."""
    noise = np.random.RandomState(seed).normal(
        0.0, _QUANTILE_JITTER_STD, x_train.shape
    ).astype(x_train.dtype)
    return x_train + noise


def fit_quantile_normal_transformer(
    x_train: np.ndarray, seed: int
) -> sklearn.preprocessing.QuantileTransformer:
    """Fit a normal-output ``QuantileTransformer`` with the standard jitter recipe."""
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_train.shape[0] // 30, 1000), 10),
        output_distribution="normal",
        subsample=1_000_000_000,
        random_state=seed,
    )
    transformer.fit(add_quantile_jitter(x_train, seed))
    return transformer


def column_centers(
    train: np.ndarray, center_fn: Callable[..., np.ndarray] = np.nanmean
) -> np.ndarray:
    """Per-column center (nanmean/nanmedian) with all-NaN columns set to 0."""
    centers = center_fn(train, axis=0)
    return np.where(np.isnan(centers), 0.0, centers)


def fill_nans(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(values)
    if not nan_mask.any():
        return values
    return np.where(nan_mask, centers, values)


T = TypeVar("T", bound=np.ndarray)


def filter_constant_columns(
    parts: dict[str, T],
) -> tuple[dict[str, T], np.ndarray]:
    """Drop columns that are constant on the train split. Returns ``(parts, keep_mask)``."""
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in parts["train"].T], dtype=bool
    )
    return {part: values[:, keep_mask] for part, values in parts.items()}, keep_mask


__all__ = [
    "signed_log1p",
    "signed_expm1",
    "add_quantile_jitter",
    "fit_quantile_normal_transformer",
    "column_centers",
    "fill_nans",
    "filter_constant_columns",
]
