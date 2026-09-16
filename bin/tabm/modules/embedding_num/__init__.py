from .advanced_plr import (
    build_num_embedding_v4,
    build_num_embedding_v5,
    build_num_embedding_v6,
    build_num_embedding_v7,
    build_num_embedding_v8,
)
from .gated_plr import build_num_embedding_v1
from .piecewise_linear import build_num_embedding_v0
from .polynomial import build_num_embedding_v2
from .tokenized import build_num_embedding_v3

NUM_EMBEDDING_MAP = {
    0: build_num_embedding_v0,
    1: build_num_embedding_v1,
    2: build_num_embedding_v2,
    3: build_num_embedding_v3,
    4: build_num_embedding_v4,
    5: build_num_embedding_v5,
    6: build_num_embedding_v6,
    7: build_num_embedding_v7,
    8: build_num_embedding_v8,
}

__all__ = [
    "build_num_embedding_v0",
    "build_num_embedding_v1",
    "build_num_embedding_v2",
    "build_num_embedding_v3",
    "build_num_embedding_v4",
    "build_num_embedding_v5",
    "build_num_embedding_v6",
    "build_num_embedding_v7",
    "build_num_embedding_v8",
    "NUM_EMBEDDING_MAP",
]
