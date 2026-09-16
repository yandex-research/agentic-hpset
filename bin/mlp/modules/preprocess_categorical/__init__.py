from .categorical import categorical_preprocess_v0
from .categorical_hash import categorical_preprocess_v1
from .categorical_rarebucket import categorical_preprocess_v2
from .categorical_targetorder import categorical_preprocess_v3

CAT_PREPROCESS_MAP = {
    0: categorical_preprocess_v0,
    1: categorical_preprocess_v1,
    2: categorical_preprocess_v2,
    3: categorical_preprocess_v3,
}

__all__ = [
    "categorical_preprocess_v0",
    "categorical_preprocess_v1",
    "categorical_preprocess_v2",
    "categorical_preprocess_v3",
    "CAT_PREPROCESS_MAP",
]
