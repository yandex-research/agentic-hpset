from .large_cat_frequency import build_categorical_large_cat_frequency
from .rare_merge import build_categorical_rare_merge
from .realmlp import build_categorical_v0


CAT_PREPROCESS_MAP = {
    0: build_categorical_v0,
    1: build_categorical_rare_merge,
    2: build_categorical_large_cat_frequency,
}

__all__ = [
    "build_categorical_v0",
    "build_categorical_rare_merge",
    "build_categorical_large_cat_frequency",
    "CAT_PREPROCESS_MAP",
]
