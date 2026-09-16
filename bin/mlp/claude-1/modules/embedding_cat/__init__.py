"""Static registry: one local file per distinct implementation."""

from .baseline import build_cat_embedding_v0 as _v0
from .clf_learned_v0 import build_cat_embedding_v1 as _v1
from .clf_hashed_v0 import build_cat_embedding_v2 as _v2
from .reg_learned_dense import build_cat_embedding_learned_dense as _v3
from .reg_hash import build_cat_embedding_hash as _v4

CAT_EMBEDDING_MAP = {
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
