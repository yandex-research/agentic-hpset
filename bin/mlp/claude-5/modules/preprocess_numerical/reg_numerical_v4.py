# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v4``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing
from ...core import bounded_numerical_parts


def numerical_preprocess_v4(
    x_num: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None})
    bounded, support = bounded_numerical_parts(x_num)
    transformer = sklearn.preprocessing.PowerTransformer(
        method='yeo-johnson', standardize=True
    ).fit(bounded['train'])
    transformed = {
        part: np.nan_to_num(transformer.transform(values)).astype(np.float32)
        for part, values in bounded.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (
            None,
            {'transformer': transformer, 'keep_mask': keep_mask, **support},
        )
    return (
        transformed,
        {'transformer': transformer, 'keep_mask': keep_mask, **support},
    )
