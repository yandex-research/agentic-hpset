"""Static registry: one local file per distinct implementation."""

from .baseline import build_cat_embedding_v0 as _v0
from .clf_learned_dense import build_cat_embedding_v1 as _v1
from .clf_target_encoded import build_cat_embedding_v3 as _v2
from .clf_freq_concat import build_cat_embedding_v4 as _v3
from .reg_learned import build_cat_embedding_v1 as _v4
from .reg_hashing import build_cat_embedding_v2 as _v5
from .reg_shared_table import build_cat_embedding_v3 as _v6

CAT_EMBEDDING_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
    6: _v6,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3},
    'regression': {0, 4, 5, 6},
}
ALIASES: dict[int, int] = {}
