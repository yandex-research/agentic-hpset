# ruff: noqa
"""Standalone implementation for ``numerical_quantile_plus_raw_v2``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _drop_constant(
    values: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray] | None, np.ndarray]:
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in values['train'].T], dtype=bool
    )
    values = {
        part: part_values[:, keep_mask].astype(np.float32)
        for part, part_values in values.items()
    }
    if values['train'].shape[1] == 0:
        return (None, keep_mask)
    return (values, keep_mask)


def _impute_with_train_median(
    x_num: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    train = x_num['train'].astype(np.float32)
    medians = np.nanmedian(train, axis=0).astype(np.float32)
    medians = np.nan_to_num(medians, nan=0.0).astype(np.float32)
    return (
        {
            part: np.where(np.isnan(values), medians, values).astype(np.float32)
            for part, values in x_num.items()
        },
        medians,
    )


def numerical_preprocess_v0(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v0 requires a seed in config.')
    x_num_train = x_num['train']
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_num_train.shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    noisy_train = x_num_train + np.random.RandomState(seed).normal(
        0.0, 1e-05, x_num_train.shape
    ).astype(x_num_train.dtype)
    transformer.fit(noisy_train)
    transformed = {
        part: transformer.transform(values) for part, values in x_num.items()
    }
    transformed = {
        part: np.nan_to_num(values).astype(np.float32)
        for part, values in transformed.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'transformer': transformer, 'keep_mask': keep_mask})
    return (transformed, {'transformer': transformer, 'keep_mask': keep_mask})


def numerical_quantile_plus_raw_v2(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'base': None, 'medians': None, 'iqr': None})
    base, artifacts = numerical_preprocess_v0(x_num, config)
    imputed, medians = _impute_with_train_median(x_num)
    q1 = np.nanpercentile(imputed['train'], 25, axis=0)
    q3 = np.nanpercentile(imputed['train'], 75, axis=0)
    iqr = np.where(q3 - q1 > 1e-06, q3 - q1, 1.0).astype(np.float32)
    raw_scaled = {
        part: np.clip((values - medians) / iqr, -8.0, 8.0).astype(np.float32)
        for part, values in imputed.items()
    }
    raw_scaled, raw_keep_mask = _drop_constant(raw_scaled)
    if base is None:
        return (
            raw_scaled,
            {
                'base': artifacts,
                'medians': medians,
                'iqr': iqr,
                'raw_keep_mask': raw_keep_mask,
            },
        )
    if raw_scaled is None:
        return (
            base,
            {
                'base': artifacts,
                'medians': medians,
                'iqr': iqr,
                'raw_keep_mask': raw_keep_mask,
            },
        )
    return (
        {
            part: np.column_stack([base[part], raw_scaled[part]]).astype(np.float32)
            for part in base
        },
        {
            'base': artifacts,
            'medians': medians,
            'iqr': iqr,
            'raw_keep_mask': raw_keep_mask,
        },
    )
