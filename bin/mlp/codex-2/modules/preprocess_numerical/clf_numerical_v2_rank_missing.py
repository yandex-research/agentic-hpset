# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _median_impute(values: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, center).astype(np.float32)


def numerical_preprocess_v2(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (
            None,
            {
                'transformer': None,
                'center': None,
                'missing_columns': None,
                'keep_mask': None,
            },
        )
    config = config or {}
    seed = int(config.get('seed', 0))
    train = np.asarray(x_num['train'], dtype=np.float32)
    center = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0).astype(
        np.float32
    )
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    missing_columns = np.isnan(train).any(axis=0)
    imputed_train = _median_impute(train, center)
    noise = (
        np.random.RandomState(seed)
        .normal(0.0, 1e-06, imputed_train.shape)
        .astype(np.float32)
    )
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(imputed_train.shape[0], 1000), 10),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    ).fit(imputed_train + noise)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        values = np.asarray(values, dtype=np.float32)
        ranked = transformer.transform(_median_impute(values, center))
        ranked = np.nan_to_num(ranked, nan=0.0, posinf=8.0, neginf=-8.0).astype(
            np.float32
        )
        pieces = [np.clip(ranked, -8.0, 8.0)]
        if missing_columns.any():
            pieces.append(np.isnan(values).astype(np.float32)[:, missing_columns])
        transformed[part] = np.column_stack(pieces).astype(np.float32)
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {
                'transformer': transformer,
                'center': center,
                'missing_columns': missing_columns,
                'keep_mask': keep_mask,
            },
        )
    return (
        transformed,
        {
            'transformer': transformer,
            'center': center,
            'missing_columns': missing_columns,
            'keep_mask': keep_mask,
        },
    )
