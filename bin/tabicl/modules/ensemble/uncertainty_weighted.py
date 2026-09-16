from __future__ import annotations

import numpy as np

from bin.tabicl.core import is_classification_task

from ._base import normalize_proba


def aggregate_uncertainty_weighted(
    per_member, *, task_type, n_classes=None, uncertainty=None
):
    if is_classification_task(task_type):
        out = {}
        for part, preds in per_member.items():
            if uncertainty is None:
                entropy = -np.sum(
                    np.clip(preds, 1e-12, 1.0) * np.log(np.clip(preds, 1e-12, 1.0)),
                    axis=-1,
                )
            else:
                entropy = uncertainty[part]
            weights = 1.0 / (entropy + 1e-3)
            weights = weights / weights.sum(axis=0, keepdims=True)
            out[part] = normalize_proba(np.sum(preds * weights[..., None], axis=0))
        return out
    if uncertainty is None:
        return {
            part: preds.mean(axis=0).astype(np.float32)
            for part, preds in per_member.items()
        }
    out = {}
    for part, preds in per_member.items():
        spread = np.maximum(uncertainty[part], 0.0)
        weights = 1.0 / (spread + 1e-3)
        weights = weights / weights.sum(axis=0, keepdims=True)
        out[part] = np.sum(preds * weights, axis=0).astype(np.float32)
    return out


__all__ = ["aggregate_uncertainty_weighted"]
