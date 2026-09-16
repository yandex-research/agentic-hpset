"""Static registry: one local file per distinct implementation."""

from .baseline import numerical_preprocess_v0 as _v0
from .clf_numerical_yeo_johnson import numerical_preprocess_v1 as _v1
from .clf_numerical_with_outliers import numerical_preprocess_v2 as _v2
from .clf_numerical_rank_concat import numerical_preprocess_v3 as _v3
from .reg_numerical_v1 import numerical_preprocess_v1 as _v4
from .reg_numerical_v3 import numerical_preprocess_v3 as _v5
from .reg_numerical_v4 import numerical_preprocess_v4 as _v6
from .reg_numerical_v5 import numerical_preprocess_v5 as _v7

NUM_PREPROCESS_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
    6: _v6,
    7: _v7,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3},
    'regression': {0, 4, 5, 6, 7},
}
ALIASES: dict[int, int] = {}
