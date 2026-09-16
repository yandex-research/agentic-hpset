from .piecewise_linear import build_num_embedding_v0
from .periodic import build_num_embedding_v1
from .linear import build_num_embedding_v2
from .bspline import build_num_embedding_v3
from .bilinear import build_num_embedding_v5
from .plr_tree import build_num_embedding_v6
from .plr_learnable import build_num_embedding_v7
from .plr_perfeat_mlp import build_num_embedding_v8
from .soft_binning import build_num_embedding_v10
from .plr_normed import build_num_embedding_v11
from .plr_fm import build_num_embedding_v12
from .plr_rank import build_num_embedding_v13

NUM_EMBEDDING_MAP = {
    0: build_num_embedding_v0,
    1: build_num_embedding_v1,
    2: build_num_embedding_v2,
    3: build_num_embedding_v3,
    5: build_num_embedding_v5,
    6: build_num_embedding_v6,
    7: build_num_embedding_v7,
    8: build_num_embedding_v8,
    10: build_num_embedding_v10,
    11: build_num_embedding_v11,
    12: build_num_embedding_v12,
    13: build_num_embedding_v13,
}

__all__ = [
    "build_num_embedding_v0",
    "build_num_embedding_v1",
    "build_num_embedding_v2",
    "build_num_embedding_v3",
    "build_num_embedding_v5",
    "build_num_embedding_v6",
    "build_num_embedding_v7",
    "build_num_embedding_v8",
    "build_num_embedding_v10",
    "build_num_embedding_v11",
    "build_num_embedding_v12",
    "build_num_embedding_v13",
    "NUM_EMBEDDING_MAP",
]
