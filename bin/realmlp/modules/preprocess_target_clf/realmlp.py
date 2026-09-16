"""RealMLP classifier target preprocessing (v0).

Wrap sklearn ``OrdinalEncoder`` for the label column. Returns the integer-encoded
labels, the recovered ``classes_`` array, and an ``inverse_transform`` closure
that maps integer predictions back to the original label space.

Applied **once per fit call** at the wrapper level (not per ensemble member).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def target_preprocess_clf_v0(y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Encode label columns to int64 indices.

    ``y_raw`` is the 2D label matrix (n_samples, 1) — the wrapper has already
    coerced 1D inputs.
    """
    encoder = OrdinalEncoder(dtype=np.int64)
    y_fit = encoder.fit_transform(pd.DataFrame(y_raw))
    classes = encoder.categories_[0]
    y_int = y_fit[:, 0].astype(np.int64)

    def inverse(idx: np.ndarray) -> np.ndarray:
        return np.asarray(classes)[idx]

    return y_int, classes, inverse

__all__ = ["target_preprocess_clf_v0"]
