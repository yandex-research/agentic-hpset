from __future__ import annotations

from ._base import RegressionPostproc, RegressionTargetTransform
from .log1p import regression_postproc_log1p
from .standardize import regression_postproc_standardize

# index -> RegressionPostproc instance (stateless; carries the log flag).
# Indices 2/3 (transductive variants) were removed for leaking val into the
# model context; gaps are intentional -- never renumber survivors.
REGRESSION_POSTPROC_MAP = {
    0: regression_postproc_standardize,
    1: regression_postproc_log1p,
}
REGRESSION_POSTPROC_BASKETS = {
    0: (0,),
    1: (0, 1),
}
REGRESSION_POSTPROC_NAMES = {
    0: "standardize",
    1: "log1p",
}

__all__ = [
    "REGRESSION_POSTPROC_MAP",
    "REGRESSION_POSTPROC_BASKETS",
    "REGRESSION_POSTPROC_NAMES",
    "RegressionPostproc",
    "RegressionTargetTransform",
    "regression_postproc_standardize",
    "regression_postproc_log1p",
]
