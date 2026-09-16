from __future__ import annotations

from ._base import FeatureEncoder, FeaturePreprocessor
from .identity import feature_preproc_identity
from .power import feature_preproc_power
from .quantile_normal import feature_preproc_quantile_normal
from .quantile_uniform import feature_preproc_quantile_uniform
from .raw_quantile_normal import feature_preproc_raw_quantile_normal

# index -> factory(*, outlier_threshold, random_state) -> FeaturePreprocessor
FEATURE_PREPROC_MAP = {
    0: feature_preproc_identity,
    1: feature_preproc_power,
    2: feature_preproc_quantile_normal,
    3: feature_preproc_quantile_uniform,
    4: feature_preproc_raw_quantile_normal,
}
# basket index -> tuple of feature-preproc indices mixed across ensemble members
FEATURE_PREPROC_BASKETS = {
    0: (0, 1),
    1: (0, 1, 2, 3),
    2: (0, 1, 4),
}
FEATURE_PREPROC_NAMES = {
    0: "identity",
    1: "power",
    2: "quantile_normal",
    3: "quantile_uniform",
    4: "raw_quantile_normal",
}

__all__ = [
    "FEATURE_PREPROC_MAP",
    "FEATURE_PREPROC_BASKETS",
    "FEATURE_PREPROC_NAMES",
    "FeatureEncoder",
    "FeaturePreprocessor",
    "feature_preproc_identity",
    "feature_preproc_power",
    "feature_preproc_quantile_normal",
    "feature_preproc_quantile_uniform",
    "feature_preproc_raw_quantile_normal",
]
