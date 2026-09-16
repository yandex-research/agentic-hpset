# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1_rare_fold``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def categorical_preprocess_v1_rare_fold(
    x_cat: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'encoder': None, 'rare_token': None, 'min_count': None})
    config = config or {}
    min_count = int(config.get('rare_min_count', 5))
    rare_marker = '__RARE__'
    folded_train = []
    folded_val = []
    folded_test = []
    train_arr = x_cat['train']
    val_arr = x_cat['val']
    test_arr = x_cat['test']
    for col in range(train_arr.shape[1]):
        train_col = np.asarray(train_arr[:, col], dtype=str).astype(object)
        val_col = np.asarray(val_arr[:, col], dtype=str).astype(object)
        test_col = np.asarray(test_arr[:, col], dtype=str).astype(object)
        values, counts = np.unique(train_col, return_counts=True)
        rare = set(values[counts < min_count].tolist())
        if rare:
            train_col = np.where(np.isin(train_col, list(rare)), rare_marker, train_col)
            val_col = np.where(np.isin(val_col, list(rare)), rare_marker, val_col)
            test_col = np.where(np.isin(test_col, list(rare)), rare_marker, test_col)
        folded_train.append(train_col)
        folded_val.append(val_col)
        folded_test.append(test_col)
    train_folded = np.column_stack(folded_train) if folded_train else train_arr
    val_folded = np.column_stack(folded_val) if folded_val else val_arr
    test_folded = np.column_stack(folded_test) if folded_test else test_arr
    unknown_value = np.iinfo(np.int64).max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=unknown_value, dtype=np.int64
    ).fit(train_folded)
    encoded = {
        'train': encoder.transform(train_folded).astype(np.int64),
        'val': encoder.transform(val_folded).astype(np.int64),
        'test': encoder.transform(test_folded).astype(np.int64),
    }
    train_max = (
        encoded['train'].max(axis=0)
        if encoded['train'].shape[1]
        else np.array([], dtype=np.int64)
    )
    for part in ('val', 'test'):
        for column in range(encoded[part].shape[1]):
            mask = encoded[part][:, column] == unknown_value
            if mask.any():
                encoded[part][mask, column] = train_max[column] + 1
    return (
        encoded,
        {'encoder': encoder, 'rare_token': rare_marker, 'min_count': min_count},
    )
