"""Static registry: one local file per distinct implementation."""

from .baseline import categorical_preprocess_v0 as _v0
from .clf_categorical_v1_rare_bucket import categorical_preprocess_v1 as _v1
from .clf_categorical_v2_frequency_order import categorical_preprocess_v2 as _v2
from .clf_categorical_v3_top_frequency import categorical_preprocess_v3 as _v3
from .clf_categorical_v5_hash_cap import categorical_preprocess_v5 as _v4
from .reg_categorical_frequency_order import categorical_preprocess_v1 as _v5
from .reg_categorical_rare_bucket import categorical_preprocess_v2 as _v6
from .reg_categorical_count_bins import categorical_preprocess_v3 as _v7
from .reg_categorical_dual_rare import categorical_preprocess_v4 as _v8

CAT_PREPROCESS_MAP = {
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
