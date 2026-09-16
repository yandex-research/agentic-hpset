"""Static registry: one local file per distinct implementation."""

from .baseline import build_cat_embedding_v0 as _v0
from .clf_entity import build_cat_embedding_v1 as _v1
from .clf_entity_dropout import build_cat_embedding_v2 as _v2
from .clf_cardinality_scaled import build_cat_embedding_v3 as _v3
from .clf_hashed_one_hot import build_cat_embedding_v4 as _v4
from .reg_cat_embedding_v3 import build_cat_embedding_v3 as _v5

CAT_EMBEDDING_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4},
    'regression': {0, 5},
}
ALIASES: dict[int, int] = {}
