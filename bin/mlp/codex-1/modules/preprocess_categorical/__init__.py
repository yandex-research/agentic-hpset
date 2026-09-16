"""Static registry: one local file per distinct implementation."""

from .baseline import categorical_preprocess_v0 as _v0
from .clf_categorical_v1 import categorical_preprocess_v1 as _v1
from .clf_categorical_v4 import categorical_preprocess_v4 as _v2
from .reg_categorical_rare import categorical_preprocess_v1 as _v3
from .reg_categorical_rare_missing import categorical_preprocess_v3 as _v4

CAT_PREPROCESS_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
}
APPLICABLE = {
    'classification': {0, 1, 2},
    'regression': {0, 3, 4},
}
ALIASES: dict[int, int] = {}
