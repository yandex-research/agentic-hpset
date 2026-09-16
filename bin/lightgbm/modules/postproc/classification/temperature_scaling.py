"""Mild temperature scaling fitted on training log-loss.

Searches `T` on a tight grid `{0.8, 0.9, 1.0, 1.1, 1.2}`, applies it
to the training logits via `softmax(logits / T)`, and picks the `T`
with the lowest training log loss. The same `T` is then applied to
validation and test predictions.
"""

from __future__ import annotations

import numpy as np

_T_GRID = (0.8, 0.9, 1.0, 1.1, 1.2)
_EPS = 1e-6


def _binarize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.column_stack([1.0 - arr, arr])
    return arr


def _apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    safe = np.clip(p, _EPS, 1.0 - _EPS)
    logits = np.log(safe)
    scaled = logits / max(t, _EPS)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    y_int = np.asarray(y, dtype=np.int64).reshape(-1)
    n, k = p.shape
    if y_int.size != n or y_int.min() < 0 or y_int.max() >= k:
        return float("inf")
    rows = np.arange(n)
    chosen = np.clip(p[rows, y_int], _EPS, 1.0)
    return float(-np.log(chosen).mean())


def temperature_scaling_postproc(
    proba: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    train_proba = _binarize(np.asarray(proba["train"], dtype=np.float64))
    best_t = 1.0
    best_loss = _log_loss(train_proba, y_train)
    for t in _T_GRID:
        if t == 1.0:
            continue
        candidate = _apply_temperature(train_proba, t)
        loss = _log_loss(candidate, y_train)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    out: dict[str, np.ndarray] = {}
    for part, arr in proba.items():
        a = _binarize(np.asarray(arr, dtype=np.float64))
        out[part] = _apply_temperature(a, best_t).astype(np.float32)
    return out
