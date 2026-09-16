"""Static registry: one local file per distinct implementation."""

from .baseline import numerical_preprocess_v0 as _v0
from .clf_numerical_nan_indicator import numerical_preprocess_v3 as _v1
from .reg_numerical_rank_gauss import numerical_preprocess_v1 as _v2
from .reg_numerical_yeo_johnson import numerical_preprocess_v2 as _v3

NUM_PREPROCESS_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
}
APPLICABLE = {
    'classification': {0, 1},
    'regression': {0, 2, 3},
}
ALIASES: dict[int, int] = {}
