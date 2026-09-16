"""Top-K PCA components of the numeric feature block.

Fits PCA on a bounded subsample of train numeric columns (median-imputed)
and transforms every split. Adds up to 8 numeric components. Useful when
the raw numeric block has redundant linear structure that splits can't
recover quickly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from bin.lightgbm.core import subsample_indices

_N_COMPONENTS = 8
_FIT_LIMIT = 50_000
_PREFIX = "pca"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def pca_components_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    num_cols = _numeric_cols(x["train"])
    if len(num_cols) < 2:
        return out

    train_block = (
        x["train"][num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    )
    medians = np.nanmedian(train_block, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train_block = np.where(np.isnan(train_block), medians, train_block)

    n = train_block.shape[0]
    fit_idx = subsample_indices(n, _FIT_LIMIT, seed=0)
    n_components = min(_N_COMPONENTS, train_block.shape[1], fit_idx.size)
    if n_components < 1:
        return out

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(train_block[fit_idx])

    for part, frame in out.items():
        block = (
            frame[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        )
        block = np.where(np.isnan(block), medians, block)
        transformed = pca.transform(block).astype(np.float32)
        for i in range(transformed.shape[1]):
            out[part][f"{_PREFIX}_{i}"] = transformed[:, i]
    return out
