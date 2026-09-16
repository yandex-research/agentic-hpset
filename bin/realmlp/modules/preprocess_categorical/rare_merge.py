"""Categorical preprocessing (v1): merge rare categories into the unknown bucket.

Hypothesis: v0 gives every category seen in training its own one-hot column /
embedding row, so long-tail categories with a handful of occurrences get pure
noise parameters. Min-frequency grouping — a first-class sklearn
``OneHotEncoder`` feature and standard practice in strong tabular pipelines —
remaps categories with train count < ``cat_min_frequency`` (default 5) into
the same index-0 bucket that unseen/missing values use, pooling them with the
unknown prior and shrinking one-hot blocks and embedding tables.

Everything else mirrors v0: ordinal encoding with unknown -> 0, the
``max_one_hot_cat_size`` small/large split, and the size-2 / size-3 one-hot
compressions (via the shared v0 helpers). Surviving category indices are
re-compacted to stay contiguous.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from .realmlp import _encode_small_one_hot


def _remap_columns(cat_all: np.ndarray, remaps: List[np.ndarray]) -> np.ndarray:
    if cat_all.shape[1] == 0:
        return cat_all
    out = np.empty_like(cat_all)
    for j, remap in enumerate(remaps):
        col = cat_all[:, j]
        # Values beyond the train-time table (defensive; encoder should not
        # produce them) collapse to the unknown bucket.
        col = np.where((col >= 0) & (col < len(remap)), col, 0)
        out[:, j] = remap[col]
    return out


def build_categorical_rare_merge(cfg: Dict[str, Any]) -> Dict[str, Callable[..., Any]]:
    """Return ``{fit, encode_small_onehot, encode_large_indices}`` mirroring v0."""
    max_one_hot = int(cfg.get("max_one_hot_cat_size", 9))
    min_count = int(cfg.get("cat_min_frequency", 5))

    def fit(X_cat_df: pd.DataFrame) -> Dict[str, Any]:
        if X_cat_df.shape[1] == 0:
            return {
                "encoder": None,
                "remaps": [],
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
        raw_sizes = (cat_all.max(axis=0) + 1).astype(int).tolist()
        remaps: List[np.ndarray] = []
        cat_sizes: List[int] = []
        for j, raw_size in enumerate(raw_sizes):
            counts = np.bincount(cat_all[:, j], minlength=raw_size)
            remap = np.zeros(raw_size, dtype=np.int64)
            next_idx = 1
            for value in range(1, raw_size):
                if counts[value] >= min_count:
                    remap[value] = next_idx
                    next_idx += 1
            remaps.append(remap)
            cat_sizes.append(next_idx)
        small_idxs = [i for i, s in enumerate(cat_sizes) if s <= max_one_hot]
        large_idxs = [i for i, s in enumerate(cat_sizes) if s > max_one_hot]
        return {
            "encoder": encoder,
            "remaps": remaps,
            "cat_sizes": cat_sizes,
            "small_idxs": small_idxs,
            "large_idxs": large_idxs,
            "large_sizes": [cat_sizes[i] for i in large_idxs],
            "categories": [list(cats) for cats in encoder.categories_],
        }

    def _transform_merged(X_cat_df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
        if artifacts["encoder"] is None or X_cat_df.shape[1] == 0:
            return np.zeros((len(X_cat_df), 0), dtype=np.int64)
        cat_all = (artifacts["encoder"].transform(X_cat_df) + 1).astype(np.int64)
        return _remap_columns(cat_all, artifacts["remaps"])

    def encode_small_onehot(X_cat_df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
        return _encode_small_one_hot(_transform_merged(X_cat_df, artifacts), artifacts)

    def encode_large_indices(X_cat_df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
        cat_all = _transform_merged(X_cat_df, artifacts)
        if not artifacts["large_idxs"]:
            return np.zeros((cat_all.shape[0], 0), dtype=np.int64)
        return cat_all[:, artifacts["large_idxs"]].astype(np.int64)

    return {
        "fit": fit,
        "encode_small_onehot": encode_small_onehot,
        "encode_large_indices": encode_large_indices,
    }


__all__ = ["build_categorical_rare_merge"]
