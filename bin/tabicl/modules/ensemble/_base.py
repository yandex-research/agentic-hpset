from __future__ import annotations

import numpy as np


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    return (proba / proba.sum(axis=-1, keepdims=True)).astype(np.float32)


__all__ = ["normalize_proba"]
