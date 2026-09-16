"""RealMLP regression target preprocessing (v0).

Applied **per ensemble member**: uses the member's train slice to compute
``y_mean`` / ``y_std`` (for normalization) and ``y_min`` / ``y_max`` (used by
``clamp_output`` at inference time).

When ``normalize_output=True`` (default), training and validation targets are
standardized by ``(y - y_mean) / (y_std + 1e-30)``. Both ``y_mean`` and
``y_std`` end up in the member's artifacts so the inference stage can undo the
scaling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

def target_preprocess_reg_v0(
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Optional[np.ndarray]]]:
    y_train = y_train.astype(np.float32, copy=True)
    y_val = y_val.astype(np.float32, copy=True) if y_val.size else y_val
    y_min = y_train.min(axis=0)
    y_max = y_train.max(axis=0)
    if cfg.get("normalize_output", True):
        y_mean = y_train.mean(axis=0)
        y_std = y_train.std(axis=0)
        y_denom = y_std + 1e-30
        y_train = (y_train - y_mean) / y_denom
        if y_val.size:
            y_val = (y_val - y_mean) / y_denom
    else:
        y_mean = None
        y_std = None
    artifacts = {"y_mean": y_mean, "y_std": y_std, "y_min": y_min, "y_max": y_max}
    return y_train, y_val, artifacts

__all__ = ["target_preprocess_reg_v0"]
