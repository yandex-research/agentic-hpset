from .categorical import categorical_preprocess_v0
from .categorical_frequency import categorical_preprocess_v1
from .categorical_hashing import categorical_preprocess_v2

CAT_PREPROCESS_MAP = {
    0: categorical_preprocess_v0,
    1: categorical_preprocess_v1,
    2: categorical_preprocess_v2,
}

__all__ = [
    "categorical_preprocess_v0",
    "categorical_preprocess_v1",
    "categorical_preprocess_v2",
    "CAT_PREPROCESS_MAP",
]
