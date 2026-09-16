"""Static registry: one local file per distinct implementation."""

from .baseline import build_num_embedding_v0 as _v0
from .clf_tree_bins import build_num_embedding_v1 as _v1
from .clf_rbf import build_num_embedding_v2 as _v2
from .clf_learnable_bins import build_num_embedding_v4 as _v3
from .clf_rff import build_num_embedding_v5 as _v4

NUM_EMBEDDING_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4},
    'regression': {0},
}
ALIASES: dict[int, int] = {}
