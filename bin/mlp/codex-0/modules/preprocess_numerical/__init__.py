"""Static registry: one local file per distinct implementation."""

from .baseline import numerical_preprocess_v0 as _v0
from .clf_numerical_v1_standard import numerical_preprocess_v1 as _v1
from .clf_numerical_v2_robust_winsor import numerical_preprocess_v2 as _v2
from .clf_numerical_v3_power import numerical_preprocess_v3 as _v3
from .clf_numerical_v4_uniform_quantile import numerical_preprocess_v4 as _v4
from .reg_numerical_missing_indicators_v1 import numerical_missing_indicators_v1 as _v5
from .reg_numerical_quantile_plus_raw_v2 import numerical_quantile_plus_raw_v2 as _v6
from .reg_numerical_robust_winsor_v3 import numerical_robust_winsor_v3 as _v7
from .reg_numerical_top_interactions_v4 import numerical_top_interactions_v4 as _v8
from .reg_numerical_row_stats_v5 import numerical_row_stats_v5 as _v9

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
    9: _v9,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4},
    'regression': {0, 5, 6, 7, 8, 9},
}
ALIASES: dict[int, int] = {}
