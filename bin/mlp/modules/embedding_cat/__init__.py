from .one_hot import build_cat_embedding_v0
from .learned import build_cat_embedding_v1
from .freq_weighted import build_cat_embedding_v2
from .binary_coded import build_cat_embedding_v3
from .hashed import build_cat_embedding_v4
from .target_mean import build_cat_embedding_v5
from .entity import build_cat_embedding_v6
from .entity_sqrt import build_cat_embedding_v7
from .one_hot_freq import build_cat_embedding_v8

CAT_EMBEDDING_MAP = {
    0: build_cat_embedding_v0,
    1: build_cat_embedding_v1,
    2: build_cat_embedding_v2,
    3: build_cat_embedding_v3,
    4: build_cat_embedding_v4,
    5: build_cat_embedding_v5,
    6: build_cat_embedding_v6,
    7: build_cat_embedding_v7,
    8: build_cat_embedding_v8,
}

__all__ = [
    "build_cat_embedding_v0",
    "build_cat_embedding_v1",
    "build_cat_embedding_v2",
    "build_cat_embedding_v3",
    "build_cat_embedding_v4",
    "build_cat_embedding_v5",
    "build_cat_embedding_v6",
    "build_cat_embedding_v7",
    "build_cat_embedding_v8",
    "CAT_EMBEDDING_MAP",
]
