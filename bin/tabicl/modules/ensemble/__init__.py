from __future__ import annotations

from bin.tabicl.core import is_classification_task

from ._base import normalize_proba
from .geometric_mean import aggregate_geometric_mean
from .mean import aggregate_mean
from .median import aggregate_median
from .uncertainty_weighted import aggregate_uncertainty_weighted

# index -> aggregator fn(per_member, *, task_type, n_classes, uncertainty)
# Indices 3/4 (validation_weighted, greedy) were removed for fitting member
# weights on val; gaps are intentional -- never renumber survivors.
ENSEMBLE_MAP = {
    0: aggregate_mean,
    1: aggregate_median,
    2: aggregate_geometric_mean,
    5: aggregate_uncertainty_weighted,
}
ENSEMBLE_NAMES = {
    0: "mean",
    1: "median",
    2: "geometric_mean",
    5: "uncertainty_weighted",
}


def aggregate_prediction_pool(
    *,
    task_type: str,
    ensemble_idx: int,
    per_member: dict,
    n_classes: int | None = None,
    uncertainty: dict | None = None,
) -> dict:
    if ensemble_idx not in ENSEMBLE_MAP:
        raise ValueError(f"Unknown ensemble index: {ensemble_idx}")
    if is_classification_task(task_type) and n_classes is None:
        raise ValueError("n_classes is required for classification aggregation.")
    return ENSEMBLE_MAP[ensemble_idx](
        per_member,
        task_type=task_type,
        n_classes=n_classes,
        uncertainty=uncertainty,
    )


__all__ = [
    "ENSEMBLE_MAP",
    "ENSEMBLE_NAMES",
    "aggregate_prediction_pool",
    "normalize_proba",
]
