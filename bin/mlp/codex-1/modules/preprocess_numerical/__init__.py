"""Static registry: one local file per distinct implementation."""

from .baseline import numerical_preprocess_v0 as _v0
from .reg_numerical_robust_quantile import numerical_preprocess_v1 as _v1
from .reg_numerical_winsorized_standard import numerical_preprocess_v2 as _v2

NUM_PREPROCESS_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
}
APPLICABLE = {
    'classification': {0},
    'regression': {0, 1, 2},
}
ALIASES: dict[int, int] = {}
