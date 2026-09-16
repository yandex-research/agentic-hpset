# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v3(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None, 'indicator_cols': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v3 requires a seed in config.')
    x_num_raw = x_num['train'].astype(np.float64)
    nan_mask_train = np.isnan(x_num_raw)
    indicator_cols = nan_mask_train.any(axis=0)
    medians = np.where(
        nan_mask_train.all(axis=0),
        0.0,
        np.nanmedian(np.where(nan_mask_train, np.nan, x_num_raw), axis=0),
    )

    def impute(arr: np.ndarray) -> np.ndarray:
        out = arr.astype(np.float64).copy()
        mask = np.isnan(out)
        if mask.any():
            out = np.where(mask, medians, out)
        return out

    imputed = {part: impute(values) for part, values in x_num.items()}
    rng = np.random.RandomState(seed)
    noisy_train = imputed['train'] + rng.normal(0.0, 1e-05, imputed['train'].shape)
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(imputed['train'].shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    transformer.fit(noisy_train)
    transformed = {
        part: np.nan_to_num(transformer.transform(imputed[part])).astype(np.float32)
        for part in imputed
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    base = {part: values[:, keep_mask] for part, values in transformed.items()}
    if indicator_cols.any():
        indicators = {}
        for part, values in x_num.items():
            ind = np.isnan(values.astype(np.float64))[:, indicator_cols].astype(
                np.float32
            )
            indicators[part] = ind
        result = {
            part: np.concatenate([base[part], indicators[part]], axis=1)
            for part in base
        }
    else:
        result = base
    if result['train'].shape[1] == 0:
        return (
            None,
            {
                'transformer': transformer,
                'keep_mask': keep_mask,
                'indicator_cols': indicator_cols,
                'medians': medians,
            },
        )
    return (
        result,
        {
            'transformer': transformer,
            'keep_mask': keep_mask,
            'indicator_cols': indicator_cols,
            'medians': medians,
        },
    )
