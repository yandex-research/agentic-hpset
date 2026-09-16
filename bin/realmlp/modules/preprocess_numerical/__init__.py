from .quantile_normal import build_numerical_quantile_normal
from .realmlp import build_numerical_v0
from .yeo_johnson import build_numerical_yeo_johnson


NUM_PREPROCESS_MAP = {
    0: build_numerical_v0,
    1: build_numerical_quantile_normal,
    2: build_numerical_yeo_johnson,
}

__all__ = [
    "build_numerical_v0",
    "build_numerical_quantile_normal",
    "build_numerical_yeo_johnson",
    "NUM_PREPROCESS_MAP",
]
