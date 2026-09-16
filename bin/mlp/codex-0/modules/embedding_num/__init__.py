"""Static registry: one local file per distinct implementation."""

from .baseline import build_num_embedding_v0 as _v0
from .clf_piecewise_linear_v2_tail_quantile import build_num_embedding_v2 as _v1
from .clf_piecewise_linear_v3_winsor import build_num_embedding_v3 as _v2
from .reg_num_embedding_v4 import build_num_embedding_v4 as _v3

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
