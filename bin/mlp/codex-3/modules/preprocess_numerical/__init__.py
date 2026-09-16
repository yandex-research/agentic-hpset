"""Static registry: one local file per distinct implementation."""

from .baseline import numerical_preprocess_v0 as _v0
from .clf_numerical_v3_signed_log import numerical_preprocess_v3 as _v1
from .reg_numerical_robust_quantile import numerical_preprocess_v1 as _v2
from .reg_numerical_uniform_rank import numerical_preprocess_v2 as _v3
from .reg_numerical_robust_iqr import numerical_preprocess_v3 as _v4
from .reg_numerical_missing_indicators import numerical_preprocess_v5 as _v5

NUM_PREPROCESS_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
}
APPLICABLE = {
    'classification': {0, 1},
    'regression': {0, 2, 3, 4, 5},
}
ALIASES: dict[int, int] = {}
