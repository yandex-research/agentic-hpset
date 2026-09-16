from __future__ import annotations

import numpy as np

from bin.tabicl.core import is_classification_task

from ._base import normalize_proba


def aggregate_median(per_member, *, task_type, n_classes=None, uncertainty=None):
    if is_classification_task(task_type):
        return {
            part: normalize_proba(np.median(preds, axis=0))
            for part, preds in per_member.items()
        }
    return {
        part: np.median(preds, axis=0).astype(np.float32)
        for part, preds in per_member.items()
    }


__all__ = ["aggregate_median"]
