from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing

from bin.tabm.modules._shared import add_quantile_jitter, filter_constant_columns


def numerical_preprocess_v5(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return None, {"transformer": None, "keep_mask": None}
    config = config or {}
    seed = config.get("seed")
    if seed is None:
        raise ValueError("numerical_preprocess_v5 requires a seed in config.")
    # Clip inputs to the train percentile range before fitting the power
    # transform. Without this, a test feature value outside the train range is
    # extrapolated by Yeo-Johnson to a large *finite* value that the subsequent
    # standardization amplifies into a catastrophic network input (test RMSE
    # blowups on OOD test rows). Mirrors the guard in clipped_quantile (v6).
    # All-NaN columns carry no signal and cannot be fit by PowerTransformer
    # (scipy's yeo-johnson needs >=1 finite value, else it raises). Zero-fill
    # them so they become constant columns that filter_constant_columns drops
    # below, instead of crashing the whole trial.
    x_num_train = x_num["train"]
    all_nan = np.isnan(x_num_train).all(axis=0)
    if np.any(all_nan):
        x_num = {part: np.asarray(values, dtype=np.float64).copy() for part, values in x_num.items()}
        for values in x_num.values():
            values[:, all_nan] = 0.0
        x_num_train = x_num["train"]
    # NaN-safe percentiles: x_num may still contain missing values, which
    # sklearn's PowerTransformer disregards in fit and preserves in transform.
    # np.clip leaves NaN untouched, so that contract is unchanged.
    lo = np.nanpercentile(x_num_train, float(config.get("clip_low_pct", 1.0)), axis=0)
    hi = np.nanpercentile(x_num_train, float(config.get("clip_high_pct", 99.0)), axis=0)
    same = hi <= lo
    if np.any(same):
        lo = np.where(same, np.nanmin(x_num_train, axis=0), lo)
        hi = np.where(same, np.nanmax(x_num_train, axis=0), hi)
    clipped = {
        part: np.clip(values, lo, hi).astype(values.dtype)
        for part, values in x_num.items()
    }
    transformer = sklearn.preprocessing.PowerTransformer(
        method="yeo-johnson",
        standardize=False,
    )
    transformer.fit(add_quantile_jitter(clipped["train"], int(seed)))
    raw = {
        part: np.nan_to_num(
            transformer.transform(values), nan=0.0, posinf=0.0, neginf=0.0
        )
        for part, values in clipped.items()
    }
    # Yeo-Johnson is a parametric power transform: a near-degenerate column can
    # produce an extreme fitted lambda that amplifies even in-domain values, so
    # clipping the *input* range (above) is not sufficient. Clip the transformed
    # output to the train column-wise range as well, keeping val/test within the
    # exact transformed domain the model was trained on (test RMSE blowups).
    raw_lo = raw["train"].min(axis=0)
    raw_hi = raw["train"].max(axis=0)
    raw = {part: np.clip(values, raw_lo, raw_hi) for part, values in raw.items()}
    mean = raw["train"].mean(axis=0)
    std = raw["train"].std(axis=0)
    std = np.where(std == 0, 1.0, std)
    transformed = {
        part: ((values - mean) / std).astype(np.float32)
        for part, values in raw.items()
    }
    transformed, keep_mask = filter_constant_columns(transformed)
    artifacts = {
        "transformer": transformer,
        "keep_mask": keep_mask,
        "mean": mean,
        "std": std,
        "lo": lo,
        "hi": hi,
    }
    if transformed["train"].shape[1] == 0:
        return None, artifacts
    return transformed, artifacts
