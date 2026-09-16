"""Out-of-fold target-mean encoding for categorical columns.

Computes a smoothed mean target per level using a 5-fold split of train, so
each train row's encoding is computed from rows in other folds. Validation
and test use the full-train smoothed mean. Smoothing pulls rare-level means
toward the global mean to dampen leakage and variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

_PREFIX = "tgtmean"
_N_FOLDS = 5
_SMOOTH = 20.0


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        dtype = df[c].dtype
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            cols.append(c)
    return cols


def _smoothed_means(series: pd.Series, y: np.ndarray, global_mean: float) -> pd.Series:
    df = pd.DataFrame({"_k": series.values, "_y": y})
    agg = df.groupby("_k", dropna=False)["_y"].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * _SMOOTH) / (
        agg["count"] + _SMOOTH
    )
    return smoothed


def target_mean_encoding_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    cat_cols = _categorical_cols(x["train"])
    if not cat_cols:
        return out
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    global_mean = float(np.nanmean(y)) if y.size else 0.0

    n = len(x["train"])
    kf = KFold(n_splits=min(_N_FOLDS, max(2, n // 2)), shuffle=True, random_state=0)

    for col in cat_cols:
        train_keys = x["train"][col].astype("string").fillna("__nan__").to_numpy()
        oof = np.full(n, global_mean, dtype=np.float64)
        for tr_idx, va_idx in kf.split(train_keys):
            tr_series = pd.Series(train_keys[tr_idx])
            smoothed = _smoothed_means(tr_series, y[tr_idx], global_mean)
            va_series = pd.Series(train_keys[va_idx])
            oof[va_idx] = va_series.map(smoothed).fillna(global_mean).to_numpy()
        full_smoothed = _smoothed_means(
            pd.Series(train_keys), y, global_mean
        )
        new_name = f"{_PREFIX}_{col}"
        out["train"][new_name] = oof.astype(np.float32)
        for part in ("val", "test"):
            if part not in out:
                continue
            keys = out[part][col].astype("string").fillna("__nan__").to_numpy()
            mapped = pd.Series(keys).map(full_smoothed).fillna(global_mean).to_numpy()
            out[part][new_name] = mapped.astype(np.float32)
    return out
