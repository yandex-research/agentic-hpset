"""RealMLP categorical preprocessing (v0).

OrdinalEncoder fit (reserves index 0 for missing/unknown), then split features by
``max_one_hot_cat_size``:
- ``cat_size <= max_one_hot_cat_size`` -> one-hot block (with the size-2 / size-3
  compressions matching the standalone RealMLP behavior).
- ``cat_size > max_one_hot_cat_size``  -> raw int64 indices for learned embeddings.

``cat_size`` counts the missing/unknown slot, so a feature with ``c`` distinct
training values has ``cat_size = c + 1``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def _transform_all_cats(X_cat_df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
    if artifacts["encoder"] is None or X_cat_df.shape[1] == 0:
        return np.zeros((len(X_cat_df), 0), dtype=np.int64)
    return (artifacts["encoder"].transform(X_cat_df) + 1).astype(np.int64)

def _encode_small_one_hot(cat_all: np.ndarray, artifacts: Dict[str, Any]) -> np.ndarray:
    small_idxs: List[int] = artifacts["small_idxs"]
    cat_sizes: List[int] = artifacts["cat_sizes"]
    if not small_idxs:
        return np.zeros((cat_all.shape[0], 0), dtype=np.float32)
    cols = []
    for idx in small_idxs:
        values = cat_all[:, idx]
        cat_size = cat_sizes[idx]
        if cat_size == 2:
            cols.append(np.where(values == 1, 1.0, -1.0).astype(np.float32)[:, None])
        elif cat_size == 3:
            cols.append(np.choose(values, [0.0, 1.0, -1.0]).astype(np.float32)[:, None])
        else:
            one_hot = np.zeros((cat_all.shape[0], cat_size - 1), dtype=np.float32)
            known = values > 0
            if np.any(known):
                one_hot[np.arange(cat_all.shape[0])[known], values[known] - 1] = 1.0
            cols.append(one_hot)
    return np.concatenate(cols, axis=1)

def build_categorical_v0(cfg: Dict[str, Any]) -> Dict[str, Callable[..., Any]]:
    """Return ``{fit, encode_small_onehot, encode_large_indices}`` for the v0 pipeline.

    Owns ``max_one_hot_cat_size`` from ``cfg``.
    """
    max_one_hot = int(cfg.get("max_one_hot_cat_size", 9))

    def fit(X_cat_df: pd.DataFrame) -> Dict[str, Any]:
        if X_cat_df.shape[1] == 0:
            return {
                "encoder": None,
                "cat_sizes": [],
                "small_idxs": [],
                "large_idxs": [],
                "large_sizes": [],
                "categories": [],
            }
        encoder = OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        cat_all = (encoder.fit_transform(X_cat_df) + 1).astype(np.int64)
        cat_sizes = (cat_all.max(axis=0) + 1).astype(int).tolist()
        small_idxs = [i for i, s in enumerate(cat_sizes) if s <= max_one_hot]
        large_idxs = [i for i, s in enumerate(cat_sizes) if s > max_one_hot]
        return {
            "encoder": encoder,
            "cat_sizes": cat_sizes,
            "small_idxs": small_idxs,
            "large_idxs": large_idxs,
            "large_sizes": [cat_sizes[i] for i in large_idxs],
            "categories": [list(cats) for cats in encoder.categories_],
        }

    def encode_small_onehot(X_cat_df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
        cat_all = _transform_all_cats(X_cat_df, artifacts)
        return _encode_small_one_hot(cat_all, artifacts)

    def encode_large_indices(X_cat_df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
        cat_all = _transform_all_cats(X_cat_df, artifacts)
        if not artifacts["large_idxs"]:
            return np.zeros((cat_all.shape[0], 0), dtype=np.int64)
        return cat_all[:, artifacts["large_idxs"]].astype(np.int64)

    return {
        "fit": fit,
        "encode_small_onehot": encode_small_onehot,
        "encode_large_indices": encode_large_indices,
    }

__all__ = ["build_categorical_v0"]
