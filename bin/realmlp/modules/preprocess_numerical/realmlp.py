"""RealMLP numerical preprocessing (v0).

Median-centering, robust IQR scaling, smooth clipping. NaNs are not allowed.

    z = (x - median) / scale
    out = z / sqrt(1 + (z / 3)^2)

scale = IQR; falls back to 0.5 * (max - min); finally 1.0 if both are zero.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

SMOOTH_CLIP_MAX_ABS = 3.0

def _fit_center_scale(x: np.ndarray) -> Dict[str, np.ndarray]:
    if x.shape[1] == 0:
        return {"median": np.zeros(0, dtype=np.float32), "scale": np.ones(0, dtype=np.float32)}
    median = np.median(x, axis=0).astype(np.float32)
    q75 = np.quantile(x, 0.75, axis=0)
    q25 = np.quantile(x, 0.25, axis=0)
    scale = q75 - q25
    fallback = 0.5 * (np.max(x, axis=0) - np.min(x, axis=0))
    scale = np.where(scale == 0.0, fallback, scale)
    scale = np.where(scale == 0.0, 1.0, scale)
    return {"median": median, "scale": scale.astype(np.float32)}

def _apply_center_scale_clip(x: np.ndarray, artifacts: Dict[str, np.ndarray]) -> np.ndarray:
    if x.shape[1] == 0:
        return x.astype(np.float32)
    z = (x - artifacts["median"][None, :]) / artifacts["scale"][None, :]
    z = z / np.sqrt(1.0 + (z / SMOOTH_CLIP_MAX_ABS) ** 2)
    return z.astype(np.float32)

def build_numerical_v0(_cfg: Dict[str, Any]) -> Dict[str, Callable[..., Any]]:
    """Return ``{fit, transform}`` for the RealMLP-default numerical pipeline.

    fit(x_train: np.ndarray) -> artifacts
    transform(x: np.ndarray, artifacts) -> np.ndarray
    """
    def fit(x_train: np.ndarray) -> Dict[str, np.ndarray]:
        if x_train.shape[1] and np.isnan(x_train).any():
            raise ValueError("NaN values in continuous columns are currently not allowed")
        return _fit_center_scale(x_train)

    def transform(x: np.ndarray, artifacts: Dict[str, np.ndarray]) -> np.ndarray:
        if x.shape[1] and np.isnan(x).any():
            raise ValueError("NaN values in continuous columns are currently not allowed")
        return _apply_center_scale_clip(x, artifacts)

    return {"fit": fit, "transform": transform}

__all__ = ["build_numerical_v0"]
