"""Static registry: one local file per distinct implementation."""

from .baseline import build_cat_embedding_v0 as _v0
from .clf_learnable import build_cat_embedding_v1 as _v1
from .clf_learnable_mixed import build_cat_embedding_v2 as _v2
from .reg_learned import build_cat_embedding_v1_learned as _v3
from .reg_hash_embedding import build_cat_embedding_v2_hash as _v4
from .reg_learned_with_norm import build_cat_embedding_v3_normalized as _v5

CAT_EMBEDDING_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
}
APPLICABLE = {
    'classification': {0, 1, 2},
    'regression': {0, 3, 4, 5},
}
ALIASES: dict[int, int] = {}
