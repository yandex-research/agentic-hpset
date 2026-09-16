# ruff: noqa
"""Standalone implementation for ``numerical_robust_winsor_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np


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


def numerical_robust_winsor_v3(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'medians': None, 'iqr': None, 'lo': None, 'hi': None})
    imputed, medians = _impute_with_train_median(x_num)
    lo = np.nanpercentile(imputed['train'], 1, axis=0).astype(np.float32)
    hi = np.nanpercentile(imputed['train'], 99, axis=0).astype(np.float32)
    q1 = np.nanpercentile(imputed['train'], 25, axis=0)
    q3 = np.nanpercentile(imputed['train'], 75, axis=0)
    iqr = np.where(q3 - q1 > 1e-06, q3 - q1, 1.0).astype(np.float32)
    transformed = {
        part: ((np.clip(values, lo, hi) - medians) / iqr).astype(np.float32)
        for part, values in imputed.items()
    }
    transformed, keep_mask = _drop_constant(transformed)
    return (
        transformed,
        {'medians': medians, 'iqr': iqr, 'lo': lo, 'hi': hi, 'keep_mask': keep_mask},
    )
