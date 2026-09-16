"""Static registry: one local file per distinct implementation."""

from .baseline import build_num_embedding_v0 as _v0
from .clf_piecewise_linear_v1_equal_width import build_num_embedding_v1 as _v1
from .clf_piecewise_linear_v2_tail_quantile import build_num_embedding_v2 as _v2
from .clf_piecewise_linear_v3_winsor import build_num_embedding_v3 as _v3
from .clf_smooth_threshold_v5 import build_num_embedding_v5 as _v4
from .reg_gated_piecewise_linear_v2 import build_num_embedding_v2 as _v5
from .reg_raw_piecewise_linear_v6 import build_num_embedding_v6 as _v6

NUM_EMBEDDING_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
    6: _v6,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4},
    'regression': {0, 5, 6},
}
ALIASES: dict[int, int] = {}
