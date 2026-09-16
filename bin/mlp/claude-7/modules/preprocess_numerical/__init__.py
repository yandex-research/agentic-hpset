"""Static registry: one local file per distinct implementation."""

from .baseline import numerical_preprocess_v0 as _v0
from .clf_numerical_v1 import numerical_preprocess_v1 as _v1
from .clf_numerical_v2 import numerical_preprocess_v2 as _v2
from .clf_numerical_v3 import numerical_preprocess_v3 as _v3
from .reg_numerical_v5_rank_gauss import numerical_preprocess_v5_rank_gauss as _v4

NUM_PREPROCESS_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3},
    'regression': {0, 4},
}
ALIASES: dict[int, int] = {}
