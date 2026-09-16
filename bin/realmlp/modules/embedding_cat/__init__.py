from .realmlp import LargeCatEmbedding, build_cat_embedding_v0
from .shared_adapter import SharedCatEmbedding, build_cat_embedding_shared_adapter


CAT_EMBEDDING_MAP = {
    0: build_cat_embedding_v0,
    1: build_cat_embedding_shared_adapter,
}

__all__ = [
    "LargeCatEmbedding",
    "SharedCatEmbedding",
    "build_cat_embedding_v0",
    "build_cat_embedding_shared_adapter",
    "CAT_EMBEDDING_MAP",
]
