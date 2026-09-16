"""Per-category train-mean encoding of top-variance numeric columns.

Selects the categorical column with the smallest cardinality (capped at
256) and the top-4 numeric columns by train variance, then adds a numeric
column for each (cat, num) pair holding the train mean of `num` within the
cat level. Unseen levels in val/test fall back to the global numeric mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TOP_K_NUM = 4
_MAX_CARD = 256
_PREFIX = "grpmean"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


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


def groupby_mean_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    num_cols = _numeric_cols(x["train"])
    cat_cols = _categorical_cols(x["train"])
    if not num_cols or not cat_cols:
        return out

    cardinalities = {
        c: int(x["train"][c].astype("string").nunique(dropna=False)) for c in cat_cols
    }
    eligible_cats = [c for c, k in cardinalities.items() if k <= _MAX_CARD]
    if not eligible_cats:
        return out
    cat_col = min(eligible_cats, key=lambda c: cardinalities[c])

    variances = {
        c: float(
            np.nanvar(
                pd.to_numeric(x["train"][c], errors="coerce").to_numpy(np.float64)
            )
        )
        for c in num_cols
    }
    chosen_nums = sorted(num_cols, key=lambda c: -variances[c])[:_TOP_K_NUM]

    train_keys = x["train"][cat_col].astype("string").fillna("__nan__")
    for n_col in chosen_nums:
        vals = pd.to_numeric(x["train"][n_col], errors="coerce").to_numpy(np.float64)
        global_mean = float(np.nanmean(vals)) if vals.size else 0.0
        if not np.isfinite(global_mean):
            global_mean = 0.0
        means = pd.DataFrame({"_k": train_keys.to_numpy(), "_v": vals}).groupby(
            "_k", dropna=False
        )["_v"].mean()
        new_name = f"{_PREFIX}_{cat_col}_{n_col}"
        for part, frame in out.items():
            keys = frame[cat_col].astype("string").fillna("__nan__")
            mapped = keys.map(means).fillna(global_mean).astype(np.float32).to_numpy()
            out[part][new_name] = mapped
    return out
