from .piecewise_linear import PiecewiseLinearEmbeddings, build_num_embedding_ple
from .realmlp import PLREmbeddings, build_num_embedding_v0


NUM_EMBEDDING_MAP = {
    0: build_num_embedding_v0,
    1: build_num_embedding_ple,
}

__all__ = [
    "PLREmbeddings",
    "PiecewiseLinearEmbeddings",
    "build_num_embedding_v0",
    "build_num_embedding_ple",
    "NUM_EMBEDDING_MAP",
]
