"""Static registry: one local file per distinct implementation."""

from .baseline import numerical_preprocess_v0 as _v0
from .clf_numerical_v1_robust_missing import numerical_preprocess_v1 as _v1
from .clf_numerical_v2_rank_missing import numerical_preprocess_v2 as _v2
from .clf_numerical_v3_signed_log import numerical_preprocess_v3 as _v3
from .clf_numerical_v4_squared import numerical_preprocess_v4 as _v4
from .reg_winsorized_standard import numerical_preprocess_winsorized_standard as _v5
from .reg_robust_quantile import numerical_preprocess_robust_quantile as _v6
from .reg_rank_standard_concat import numerical_preprocess_rank_standard_concat as _v7
from .reg_yeo_johnson import numerical_preprocess_yeo_johnson as _v8

NUM_PREPROCESS_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
    6: _v6,
    7: _v7,
    8: _v8,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4},
    'regression': {0, 5, 6, 7, 8},
}
ALIASES: dict[int, int] = {}
