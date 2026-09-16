from __future__ import annotations

import numpy as np

from bin.tabicl.core import is_classification_task

from ._base import normalize_proba


def aggregate_geometric_mean(
    per_member, *, task_type, n_classes=None, uncertainty=None
):
    if not is_classification_task(task_type):
        raise ValueError("geometric_mean is classification-only.")
    return {
        part: normalize_proba(
            np.exp(np.mean(np.log(np.clip(preds, 1e-12, 1.0)), axis=0))
        )
        for part, preds in per_member.items()
    }


__all__ = ["aggregate_geometric_mean"]
