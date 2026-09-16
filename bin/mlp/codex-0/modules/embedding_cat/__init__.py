"""Static registry: one local file per distinct implementation."""

from .baseline import build_cat_embedding_v0 as _v0
from .clf_entity_v1 import build_cat_embedding_v1 as _v1
from .clf_hashed_entity_v3 import build_cat_embedding_v3 as _v2
from .clf_weighted_one_hot_v4 import build_cat_embedding_v4 as _v3
from .clf_hybrid_one_hot_entity_v5 import build_cat_embedding_v5 as _v4
from .reg_cat_embedding_v1 import build_cat_embedding_v1 as _v5
from .reg_cat_embedding_v2 import build_cat_embedding_v2 as _v6
from .reg_cat_embedding_v3 import build_cat_embedding_v3 as _v7
from .reg_cat_embedding_v4 import build_cat_embedding_v4 as _v8
from .reg_cat_embedding_v5 import build_cat_embedding_v5 as _v9

CAT_EMBEDDING_MAP = {
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
