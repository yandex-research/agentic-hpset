"""Static registry: one local file per distinct implementation."""

from .baseline import numerical_preprocess_v0 as _v0
from .clf_numerical_v2 import numerical_preprocess_v2 as _v1

NUM_PREPROCESS_MAP = {
    0: _v0,
    1: _v1,
}
APPLICABLE = {
    'classification': {0, 1},
    'regression': {0},
}
ALIASES: dict[int, int] = {}
