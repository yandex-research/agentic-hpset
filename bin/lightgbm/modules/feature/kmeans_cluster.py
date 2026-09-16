"""K-means cluster assignment and per-cluster distance features.

Fits MiniBatchKMeans with K=8 on a numeric subsample (median-imputed,
z-scaled) of train, then for every split adds: a categorical `cluster_id`
column and K numeric distance columns. Helps trees model neighborhood-style
geometry that axis-aligned splits represent poorly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from bin.lightgbm.core import subsample_indices

_K = 8
_FIT_LIMIT = 50_000
_PREFIX = "kmeans"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def kmeans_cluster_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    num_cols = _numeric_cols(x["train"])
    if not num_cols:
        return out

    train_block = (
        x["train"][num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    )
    medians = np.nanmedian(train_block, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train_block = np.where(np.isnan(train_block), medians, train_block)
    stds = np.nanstd(train_block, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    means = np.nanmean(train_block, axis=0)
    z = (train_block - means) / stds

    n = z.shape[0]
    fit_idx = subsample_indices(n, _FIT_LIMIT, seed=0)
    k = min(_K, max(2, fit_idx.size // 4))
    if k < 2:
        return out

    km = MiniBatchKMeans(
        n_clusters=k, random_state=0, batch_size=1024, n_init=3
    )
    km.fit(z[fit_idx])

    for part, frame in out.items():
        block = (
            frame[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        )
        block = np.where(np.isnan(block), medians, block)
        z_part = ((block - means) / stds).astype(np.float64)
        labels = km.predict(z_part).astype(np.int32)
        distances = km.transform(z_part).astype(np.float32)
        out[part][f"{_PREFIX}_id"] = pd.Categorical(labels)
        for i in range(distances.shape[1]):
            out[part][f"{_PREFIX}_d{i}"] = distances[:, i]
    return out
