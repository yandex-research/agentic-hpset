from .target import target_preprocess_v0
from .target_log import target_preprocess_v4
from .target_mad import target_preprocess_v3
from .target_quantile import target_preprocess_v1
from .target_symlog import target_preprocess_v2

TARGET_PREPROCESS_REG_MAP = {
    0: target_preprocess_v0,
    1: target_preprocess_v1,
    2: target_preprocess_v2,
    3: target_preprocess_v3,
    4: target_preprocess_v4,
}

__all__ = [
    "target_preprocess_v0",
    "target_preprocess_v1",
    "target_preprocess_v2",
    "target_preprocess_v3",
    "target_preprocess_v4",
    "TARGET_PREPROCESS_REG_MAP",
]
