"""Numerical preprocessing (v1): quantile transform to a standard normal.

Hypothesis: v0's linear robust scale + smooth clip compresses heavy-skewed /
heavy-tailed features into the +-3 saturation zone, losing rank resolution
where such features carry signal. An ECDF-based quantile map to N(0, 1) —
the standard preprocessing of the RTDL line (Gorishniy et al. 2021) and the
TabM pipeline — equalizes resolution across the distribution. The per-dataset
tuner selects it where it helps; on already-well-scaled features or tiny n it
can lose to v0 and is deselected.

Recipe (RTDL): break ties with small feature-scaled Gaussian noise at fit
time only, ``n_quantiles = clamp(n_train // 30, 10, 1000)``, normal output
distribution. Constant features map to 0. Output is not re-clipped: sklearn
already bounds normal outputs to roughly +-5.2.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from sklearn.preprocessing import QuantileTransformer


def build_numerical_quantile_normal(cfg: Dict[str, Any]) -> Dict[str, Callable[..., Any]]:
    """Return ``{fit, transform}`` mirroring ``build_numerical_v0``.

    fit(x_train: np.ndarray) -> artifacts
    transform(x: np.ndarray, artifacts) -> np.ndarray
    """
    noise_scale = float(cfg.get("qt_noise_scale", 1e-3))

    def fit(x_train: np.ndarray) -> Dict[str, Any]:
        if x_train.shape[1] and np.isnan(x_train).any():
            raise ValueError("NaN values in continuous columns are currently not allowed")
        if x_train.shape[1] == 0:
            return {"transformer": None}
        n = x_train.shape[0]
        transformer = QuantileTransformer(
            n_quantiles=max(min(n // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
        )
        x_fit = x_train.astype(np.float64)
        if noise_scale > 0.0:
            stds = x_fit.std(axis=0, keepdims=True)
            x_fit = x_fit + noise_scale * stds * np.random.standard_normal(x_fit.shape)
        transformer.fit(x_fit)
        return {"transformer": transformer}

    def transform(x: np.ndarray, artifacts: Dict[str, Any]) -> np.ndarray:
        if x.shape[1] and np.isnan(x).any():
            raise ValueError("NaN values in continuous columns are currently not allowed")
        if artifacts["transformer"] is None or x.shape[1] == 0:
            return x.astype(np.float32)
        return artifacts["transformer"].transform(x.astype(np.float64)).astype(np.float32)

    return {"fit": fit, "transform": transform}


__all__ = ["build_numerical_quantile_normal"]
