"""Numerical preprocessing (v2): Yeo-Johnson de-skew, then v0 robust scaling.

Hypothesis: v0 applies no de-skew, so heavily skewed features stay skewed
after linear robust scaling and saturate the smooth clip on one side. A
parametric monotone power transform (Yeo-Johnson; a first-class option in
auto-sklearn / AutoGluon preprocessing spaces, sklearn ``PowerTransformer``)
symmetrizes them and — unlike the quantile map in v1 — needs no large-n
ECDF estimate, so it also works at tiny n. After the transform the standard
v0 chain (median centering, robust IQR scaling, smooth clipping) is applied.

If the Yeo-Johnson fit fails on pathological columns, the module falls back
to plain v0 behavior for the whole block, so the harm case stays bounded.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from sklearn.preprocessing import PowerTransformer

from .realmlp import _apply_center_scale_clip, _fit_center_scale


def build_numerical_yeo_johnson(_cfg: Dict[str, Any]) -> Dict[str, Callable[..., Any]]:
    """Return ``{fit, transform}`` mirroring ``build_numerical_v0``.

    fit(x_train: np.ndarray) -> artifacts
    transform(x: np.ndarray, artifacts) -> np.ndarray
    """

    def fit(x_train: np.ndarray) -> Dict[str, Any]:
        if x_train.shape[1] and np.isnan(x_train).any():
            raise ValueError("NaN values in continuous columns are currently not allowed")
        transformer = None
        x_scaled = x_train
        if x_train.shape[1]:
            try:
                with np.errstate(all="ignore"):
                    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                    x_scaled = transformer.fit_transform(x_train.astype(np.float64))
                if not np.isfinite(x_scaled).all():
                    raise ValueError("non-finite Yeo-Johnson output")
            except Exception:
                transformer = None
                x_scaled = x_train
        return {"transformer": transformer, "center_scale": _fit_center_scale(x_scaled)}

    def transform(x: np.ndarray, artifacts: Dict[str, Any]) -> np.ndarray:
        if x.shape[1] and np.isnan(x).any():
            raise ValueError("NaN values in continuous columns are currently not allowed")
        if x.shape[1] and artifacts["transformer"] is not None:
            with np.errstate(all="ignore"):
                x = artifacts["transformer"].transform(x.astype(np.float64))
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return _apply_center_scale_clip(x, artifacts["center_scale"])

    return {"fit": fit, "transform": transform}


__all__ = ["build_numerical_yeo_johnson"]
