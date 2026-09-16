"""Categorical preprocessing (v2): add a log-frequency feature per large cat.

Hypothesis: a learned embedding row cannot see its category's global train
frequency — it only receives gradient signal proportional to it — yet
frequency itself is often predictive and is a robust statistic exactly where
per-category embeddings are noisiest (rare categories). Count/frequency
encoding is standard replicated practice in strong tabular pipelines
(category_encoders, H2O, CatBoost counters).

Mechanism: v0 encoding is kept unchanged (ordinal + unknown bucket, one-hot
small / index large). For every **large**-cardinality feature, one extra
``log1p(train count of the sample's category)`` column is appended to the
float block returned by ``encode_small_onehot``. The numeric preprocessing
stage re-fits on that block downstream, so the new columns are scaled like
every other continuous input. Unseen categories fall into the unknown bucket
and get its train count (0 if never seen -> lowest frequency).

Small-cardinality features get no frequency column: their one-hot block
already identifies the category, so the first layer can learn any frequency
effect directly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from .realmlp import build_categorical_v0


def build_categorical_large_cat_frequency(cfg: Dict[str, Any]) -> Dict[str, Callable[..., Any]]:
    """Return ``{fit, encode_small_onehot, encode_large_indices}`` mirroring v0."""
    v0 = build_categorical_v0(cfg)

    def fit(X_cat_df: pd.DataFrame) -> Dict[str, Any]:
        artifacts = v0["fit"](X_cat_df)
        counts: List[np.ndarray] = []
        if artifacts["large_idxs"]:
            cat_all = (artifacts["encoder"].transform(X_cat_df) + 1).astype(np.int64)
            for idx, size in zip(artifacts["large_idxs"], artifacts["large_sizes"]):
                counts.append(np.bincount(cat_all[:, idx], minlength=size).astype(np.int64))
        artifacts["large_counts"] = counts
        return artifacts

    def encode_small_onehot(X_cat_df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
        one_hot = v0["encode_small_onehot"](X_cat_df, artifacts)
        if not artifacts["large_idxs"]:
            return one_hot
        large = v0["encode_large_indices"](X_cat_df, artifacts)
        freq_cols = []
        for j, counts in enumerate(artifacts["large_counts"]):
            values = np.clip(large[:, j], 0, len(counts) - 1)
            freq_cols.append(np.log1p(counts[values].astype(np.float32))[:, None])
        return np.concatenate([one_hot, *freq_cols], axis=1).astype(np.float32)

    return {
        "fit": fit,
        "encode_small_onehot": encode_small_onehot,
        "encode_large_indices": v0["encode_large_indices"],
    }


__all__ = ["build_categorical_large_cat_frequency"]
