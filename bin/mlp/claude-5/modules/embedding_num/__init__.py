"""Static registry: one local file per distinct implementation."""

from .baseline import build_num_embedding_v0 as _v0
from .clf_target_aware_bins import build_num_embedding_v1 as _v1
from .clf_plr_with_raw import build_num_embedding_v4 as _v2

NUM_EMBEDDING_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
}
APPLICABLE = {
    'classification': {0, 1, 2},
    'regression': {0},
}
ALIASES: dict[int, int] = {}
