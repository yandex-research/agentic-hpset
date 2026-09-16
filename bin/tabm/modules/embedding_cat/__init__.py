from .learned import build_cat_embedding_v2
from .one_hot import build_cat_embedding_v0
from .one_hot_unknown import build_cat_embedding_v1
from .onehot_plus_learned import build_cat_embedding_v4
from .target_mean import build_cat_embedding_v3

CAT_EMBEDDING_MAP = {
    0: build_cat_embedding_v0,
    1: build_cat_embedding_v1,
    2: build_cat_embedding_v2,
    3: build_cat_embedding_v3,
    4: build_cat_embedding_v4,
}

__all__ = [
    "build_cat_embedding_v0",
    "build_cat_embedding_v1",
    "build_cat_embedding_v2",
    "build_cat_embedding_v3",
    "build_cat_embedding_v4",
    "CAT_EMBEDDING_MAP",
]
