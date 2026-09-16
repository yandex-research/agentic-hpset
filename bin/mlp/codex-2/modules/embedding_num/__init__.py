"""Static registry: one local file per distinct implementation."""

from .baseline import build_num_embedding_v0 as _v0
from .clf_piecewise_linear_v1_equal_width import build_num_embedding_v1 as _v1
from .clf_smooth_threshold_v4 import build_num_embedding_v4 as _v2
from .reg_gated_piecewise_linear import build_gated_piecewise_linear as _v3

NUM_EMBEDDING_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
}
APPLICABLE = {
    'classification': {0, 1, 2},
    'regression': {0, 3},
}
ALIASES: dict[int, int] = {}
