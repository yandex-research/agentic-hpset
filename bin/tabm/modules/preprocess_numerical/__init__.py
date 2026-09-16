from .clipped_quantile import numerical_preprocess_v6
from .missing_indicators import numerical_preprocess_v4
from .numerical import numerical_preprocess_v0
from .numerical_log1p import numerical_preprocess_v2
from .numerical_robust import numerical_preprocess_v3
from .numerical_standardize import numerical_preprocess_v1
from .quantile_kmeans import numerical_preprocess_v8
from .rank_uniform import numerical_preprocess_v7
from .yeo_johnson import numerical_preprocess_v5

NUM_PREPROCESS_MAP = {
    0: numerical_preprocess_v0,
    1: numerical_preprocess_v1,
    2: numerical_preprocess_v2,
    3: numerical_preprocess_v3,
    4: numerical_preprocess_v4,
    5: numerical_preprocess_v5,
    6: numerical_preprocess_v6,
    7: numerical_preprocess_v7,
    8: numerical_preprocess_v8,
}

__all__ = [
    "numerical_preprocess_v0",
    "numerical_preprocess_v1",
    "numerical_preprocess_v2",
    "numerical_preprocess_v3",
    "numerical_preprocess_v4",
    "numerical_preprocess_v5",
    "numerical_preprocess_v6",
    "numerical_preprocess_v7",
    "numerical_preprocess_v8",
    "NUM_PREPROCESS_MAP",
]
