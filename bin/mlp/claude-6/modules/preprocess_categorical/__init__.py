"""Static registry: one local file per distinct implementation."""

from .baseline import categorical_preprocess_v0 as _v0
from .clf_categorical_rare_collapse import categorical_preprocess_v1 as _v1
from .reg_categorical_rare import categorical_preprocess_v1 as _v2
from .reg_categorical_count import categorical_preprocess_v2 as _v3

CAT_PREPROCESS_MAP = {
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
