"""Per-category target standard deviation, encoded as a numeric feature.

For each categorical column, computes the per-level train target standard
deviation (smoothed toward the global std). Levels with high target spread
get a high value, low spread a small one. Provides a complementary signal
to mean-encoding without using the mean (less leakage potential), and
unseen levels fall back to the global std.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PREFIX = "tgtstd"
_SMOOTH = 30.0


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


def cat_target_std_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    cat_cols = _categorical_cols(x["train"])
    if not cat_cols:
        return out
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return out
    global_std = float(np.nanstd(y))
    if not np.isfinite(global_std):
        global_std = 0.0
    for col in cat_cols:
        keys = x["train"][col].astype("string").fillna("__nan__").to_numpy()
        df = pd.DataFrame({"_k": keys, "_y": y})
        agg = df.groupby("_k", dropna=False)["_y"].agg(["std", "count"]).fillna(0.0)
        smoothed = (agg["std"] * agg["count"] + global_std * _SMOOTH) / (
            agg["count"] + _SMOOTH
        )
        new_name = f"{_PREFIX}_{col}"
        for part, frame in out.items():
            keys_part = frame[col].astype("string").fillna("__nan__")
            mapped = (
                keys_part.map(smoothed).fillna(global_std).astype(np.float32).to_numpy()
            )
            out[part][new_name] = mapped
    return out
