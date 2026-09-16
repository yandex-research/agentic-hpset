from __future__ import annotations

from ._base import (
    ClassificationTargetEncoder,
    softmax,
    undo_class_permutation,
)
from .proba import ProbaPostproc

# index -> postproc class. Instantiate per member; .fit(train_logits, y_train,
# n_classes, prior) then .transform(logits) -> proba.
# Indices 1/2 (temperature, prior_blend) were removed for fitting on val;
# gaps are intentional -- never renumber survivors.
CLASSIFICATION_POSTPROC_MAP = {
    0: ProbaPostproc,
}
CLASSIFICATION_POSTPROC_BASKETS = {
    0: (0,),
}
CLASSIFICATION_POSTPROC_NAMES = {
    0: "proba",
}

__all__ = [
    "CLASSIFICATION_POSTPROC_MAP",
    "CLASSIFICATION_POSTPROC_BASKETS",
    "CLASSIFICATION_POSTPROC_NAMES",
    "ProbaPostproc",
    "ClassificationTargetEncoder",
    "softmax",
    "undo_class_permutation",
]
